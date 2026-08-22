"""Flipkart Support Capstone - Part 3 LangGraph agent."""
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from langgraph.graph import StateGraph, START, END
from build_retrieval_index import build_faiss_index

REPO_DIR = Path(__file__).resolve().parent
GROUND_THRESHOLD = 0.35
T_RF = 0.47
MOCK_LLM = True

SYSTEM_PROMPT = """
ROLE: You are Flipkart's support assistant.

4S - SPECIFIC: Answer the customer's exact request using verified policy context or actual model-tool output.
4S - SHORT: Keep the final response concise and directly useful.
4S - SURROUND: Use retrieved policy chunks and real tool outputs as evidence.
4S - SINGLE: Answer only the customer's current request.

ROLE PROMPTING: Never invent a policy that is not supported by the knowledge base.

FEW-SHOT INTENT EXAMPLE 1:
User: What is the return window for shoes?
Intent: policy

FEW-SHOT INTENT EXAMPLE 2:
User: What is the return probability for this order?
Intent: return_risk
"""

policy_chunks, embedding_model, embeddings, index = build_faiss_index()


def retrieve_policy(query: str, top_k: int = 3) -> list:
    q = embedding_model.encode([query], convert_to_numpy=True,
                               normalize_embeddings=True,
                               show_progress_bar=False).astype(np.float32)
    scores, indices = index.search(q, top_k)
    return [
        {
            "score": float(score),
            "chunk_id": policy_chunks[int(idx)]["chunk_id"],
            "parent_doc_id": policy_chunks[int(idx)]["parent_doc_id"],
            "parent_title": policy_chunks[int(idx)]["parent_title"],
            "text": policy_chunks[int(idx)]["text"],
        }
        for score, idx in zip(scores[0], indices[0])
    ]


RETURN_FEATURES = [
    "order_id", "product_category", "price_inr", "discount_pct",
    "payment_method", "customer_tenure_days", "num_previous_orders",
    "num_previous_returns", "delivery_distance_km", "delivery_days",
    "is_weekend_order", "rating_given",
]
return_risk_model = joblib.load(REPO_DIR / "models" / "return_risk_model.pkl")


def check_return_risk(order_features: dict) -> dict:
    missing = [x for x in RETURN_FEATURES if x not in order_features]
    if missing:
        raise ValueError("Missing order features: " + ", ".join(missing))
    frame = pd.DataFrame([{x: order_features[x] for x in RETURN_FEATURES}])
    probability = float(return_risk_model.predict_proba(frame)[0][1])
    high_cutoff = T_RF + 0.15
    bucket = "Low" if probability < T_RF else ("High" if probability >= high_cutoff else "Medium")
    return {"return_probability": probability, "risk_bucket": bucket, "t_rf": T_RF, "high_cutoff": high_cutoff}


checkpoint = torch.load(REPO_DIR / "models" / "product_classifier.pt", map_location="cpu")
feature_extractor = nn.Sequential(*list(models.resnet18(weights=None).children())[:-1])
feature_extractor.load_state_dict(checkpoint["feature_extractor_state_dict"])
feature_extractor.eval()
classifier_head = nn.Linear(checkpoint["feature_dimension"], checkpoint["num_classes"])
classifier_head.load_state_dict(checkpoint["classifier_head_state_dict"])
classifier_head.eval()
CLASS_NAMES = checkpoint["class_names"]
IMAGE_SIZE = checkpoint["image_size"]
IMAGENET_MEAN = checkpoint["imagenet_mean"]
IMAGENET_STD = checkpoint["imagenet_std"]


def classify_product_image(image_path: str) -> dict:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)
    tfm = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = tfm(Image.open(image_path).convert("L")).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(tensor).view(1, -1)
        probs = torch.softmax(classifier_head(features), dim=1)
        idx = int(probs.argmax(dim=1).item())
        confidence = float(probs[0, idx].item())
    return {"predicted_category": CLASS_NAMES[idx], "confidence": confidence}


class AgentState(TypedDict, total=False):
    user_message: str
    messages: List[Dict[str, str]]
    order_features: Dict[str, Any]
    image_path: str
    intent: str
    few_shot_example_used: Optional[str]
    few_shot_match_score: float
    retrieved_chunks: List[Dict[str, Any]]
    retrieval_scores: List[float]
    return_risk_result: Optional[Dict[str, Any]]
    image_classifier_result: Optional[Dict[str, Any]]
    input_blocked: bool
    grounded: bool
    guardrail_message: str
    response: Optional[Dict[str, Any]]


INJECTION_PATTERNS = [
    "ignore previous instructions", "ignore all previous instructions",
    "ignore all rules", "disregard previous instructions",
    "pretend you are", "reveal your system prompt",
]


def input_guard(state: AgentState) -> dict:
    text = state.get("user_message", "").lower()
    blocked = any(x in text for x in INJECTION_PATTERNS)
    return {
        "input_blocked": blocked,
        "guardrail_message": (
            "I can't follow instructions that attempt to override the support assistant's rules."
            if blocked else ""
        ),
    }


def intent_node(state: AgentState) -> dict:
    text = state.get(
        "user_message",
        ""
    ).lower().strip()

    # --------------------------------------------------------
    # Few-shot intent examples
    # --------------------------------------------------------

    few_shot_examples = [
        {
            "example":
                "What is the return window for shoes?",
            "intent":
                "policy",
        },
        {
            "example":
                "What is the return probability for this order?",
            "intent":
                "return_risk",
        },
    ]

    # --------------------------------------------------------
    # Deterministic few-shot matching
    # Uses token-overlap similarity so the examples actively
    # influence routing, while deterministic fallback rules
    # still handle other requests.
    # --------------------------------------------------------

    text_tokens = set(
        text.replace("?", "")
            .replace(".", "")
            .split()
    )

    best_example = None
    best_score = 0.0

    for example in few_shot_examples:

        example_tokens = set(
            example["example"]
            .lower()
            .replace("?", "")
            .replace(".", "")
            .split()
        )

        union = (
            text_tokens
            | example_tokens
        )

        intersection = (
            text_tokens
            & example_tokens
        )

        score = (
            len(intersection) / len(union)
            if union
            else 0.0
        )

        if score > best_score:
            best_score = score
            best_example = example

    # A sufficiently similar few-shot example controls routing.
    if best_example is not None and best_score >= 0.20:
        return {
            "intent":
                best_example["intent"],
            "few_shot_example_used":
                best_example["example"],
            "few_shot_match_score":
                round(best_score, 4),
        }

    # --------------------------------------------------------
    # Normal deterministic routing fallback
    # --------------------------------------------------------

    risk_words = [
        "return probability",
        "return risk",
        "risk of return",
        "chance of return",
        "likelihood of return",
    ]

    image_words = [
        "classify this product image",
        "classify this image",
        "product image",
        "product photo",
        "image classification",
    ]

    policy_words = [
        "return",
        "refund",
        "delivery",
        "pickup",
        "tracking",
        "eligible",
        "electronics",
        "apparel",
        "footwear",
        "home",
        "cod",
        "prepaid",
        "warranty",
    ]

    if any(
        x in text
        for x in risk_words
    ):
        intent = "return_risk"

    elif any(
        x in text
        for x in image_words
    ):
        intent = "product_image"

    elif any(
        x in text
        for x in policy_words
    ):
        intent = "policy"

    else:

        previous_messages = state.get(
            "messages",
            []
        )

        previous_text = " ".join(
            message.get(
                "content",
                ""
            ).lower()
            for message in previous_messages
        )

        follow_up = (
            text.startswith("and ")
            or text.startswith("what about ")
            or text.startswith("how about ")
        )

        if (
            follow_up
            and (
                "return" in previous_text
                or "refund" in previous_text
                or "delivery" in previous_text
                or "policy" in previous_text
            )
        ):
            intent = "policy"

        else:
            intent = "general"

    return {
        "intent": intent,
        "few_shot_example_used": None,
        "few_shot_match_score": 0.0,
    }

def policy_node(state: AgentState) -> dict:
    current = state.get("user_message", "").lower().strip()
    follow_up = current.startswith(("and what about ", "what about ", "how about "))
    entity = next((x for x in ["electronics", "apparel", "footwear", "home"] if x in current), None)
    query = f"What is the return policy for {entity}?" if follow_up and entity else state.get("user_message", "")
    results = retrieve_policy(query, 3)
    return {"retrieved_chunks": results, "retrieval_scores": [x["score"] for x in results]}


def risk_node(state: AgentState) -> dict:
    return {"return_risk_result": check_return_risk(state["order_features"])}


def image_node(state: AgentState) -> dict:
    return {"image_classifier_result": classify_product_image(state["image_path"])}


def mock_llm_response(state: AgentState) -> dict:
    if state.get("input_blocked"):
        return {"answer": state.get("guardrail_message", "I can't follow that instruction."), "source": "policy_kb", "confidence": 0.0}
    intent = state.get("intent", "general")
    if intent == "policy":
        scores = state.get("retrieval_scores", [])
        maximum = max(scores) if scores else 0.0
        if maximum < GROUND_THRESHOLD:
            return {"answer": "I can't verify that policy from the available policy information.", "source": "policy_kb", "confidence": 0.0}
        top = state["retrieved_chunks"][0]
        return {"answer": top["text"], "source": "policy_kb", "confidence": round(float(top["score"]), 4)}
    if intent == "return_risk":
        result = state["return_risk_result"]
        p = float(result["return_probability"])
        return {"answer": f"Estimated return probability is {p:.2%}. Risk bucket: {result['risk_bucket']}.", "source": "return_risk_tool", "confidence": p}
    if intent == "product_image":
        result = state["image_classifier_result"]
        return {"answer": f"The product image is classified as {result['predicted_category']}.", "source": "image_classifier_tool", "confidence": float(result["confidence"])}
    return {"answer": "I can help with return policy, return-risk assessment, and product-image classification.", "source": "policy_kb", "confidence": 0.0}


def response_node(state: AgentState) -> dict:
    response = mock_llm_response(state)
    if state.get("intent") == "policy" and not state.get("input_blocked", False):
        scores = state.get("retrieval_scores", [])
        return {"grounded": (max(scores) if scores else 0.0) >= GROUND_THRESHOLD, "response": response}
    return {"response": response}


def output_guard(state: AgentState) -> dict:
    response = state.get("response")
    valid_sources = {"policy_kb", "return_risk_tool", "image_classifier_tool"}
    valid = isinstance(response, dict) and all(x in response for x in ["answer", "source", "confidence"]) and response["source"] in valid_sources and isinstance(response["confidence"], (int, float))
    if not valid:
        response = {"answer": "I couldn't produce a valid verified response.", "source": "policy_kb", "confidence": 0.0}
    return {"response": response}


def route_after_guard(state: AgentState) -> str:
    return "response" if state.get("input_blocked", False) else "intent"


def route_after_intent(state: AgentState) -> str:
    return {"policy": "policy", "return_risk": "risk", "product_image": "image", "general": "response"}.get(state.get("intent", "general"), "response")


builder = StateGraph(AgentState)
builder.add_node("input_guard", input_guard)
builder.add_node("intent", intent_node)
builder.add_node("policy", policy_node)
builder.add_node("risk", risk_node)
builder.add_node("image", image_node)
builder.add_node("response", response_node)
builder.add_node("output_guard", output_guard)
builder.add_edge(START, "input_guard")
builder.add_conditional_edges("input_guard", route_after_guard, {"intent": "intent", "response": "response"})
builder.add_conditional_edges("intent", route_after_intent, {"policy": "policy", "risk": "risk", "image": "image", "response": "response"})
builder.add_edge("policy", "response")
builder.add_edge("risk", "response")
builder.add_edge("image", "response")
builder.add_edge("response", "output_guard")
builder.add_edge("output_guard", END)
final_agent = builder.compile()
