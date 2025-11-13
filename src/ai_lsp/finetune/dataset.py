import json
from pathlib import Path

from datasets import Dataset


def extract_messages_from_span(span: dict) -> list[dict] | None:
    attrs = span.get("attributes", [])
    for attr in attrs:
        if attr.get("key") == "all_messages_events":
            messages_str = attr.get("value", {}).get("stringValue", "")
            if messages_str:
                return json.loads(messages_str)
    return None


def extract_final_result_from_span(span: dict) -> dict | None:
    attrs = span.get("attributes", [])
    for attr in attrs:
        if attr.get("key") == "final_result":
            result_str = attr.get("value", {}).get("stringValue", "")
            if result_str:
                return json.loads(result_str)
    return None


def convert_otel_to_dataset(otel_file: Path) -> Dataset:
    examples = []
    
    with open(otel_file) as f:
        for line in f:
            if not line.strip():
                continue
                
            data = json.loads(line)
            resource_spans = data.get("resourceSpans", [])
            
            for resource_span in resource_spans:
                scope_spans = resource_span.get("scopeSpans", [])
                
                for scope_span in scope_spans:
                    if scope_span.get("scope", {}).get("name") != "pydantic-ai":
                        continue
                    
                    spans = scope_span.get("spans", [])
                    for span in spans:
                        if span.get("name") != "agent run":
                            continue
                        
                        messages = extract_messages_from_span(span)
                        final_result = extract_final_result_from_span(span)
                        
                        if not messages or not final_result:
                            continue
                        
                        system_msg = None
                        user_msg = None
                        
                        for msg in messages:
                            role = msg.get("role")
                            content = msg.get("content")
                            
                            if role == "system":
                                system_msg = content
                            elif role == "user":
                                user_msg = content
                        
                        if system_msg and user_msg and final_result:
                            examples.append({
                                "system": system_msg,
                                "user": user_msg,
                                "assistant": json.dumps(final_result, indent=2),
                            })
    
    return Dataset.from_list(examples)
