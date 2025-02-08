"""Core template system for generating questions."""

from typing import Dict, Any, Callable
from dataclasses import dataclass
import random

def execute_template(template_str: str, parts: Dict[str, str | Callable], refs: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a template with its parts and references."""
    values = {}
    executed_parts = {}

    for name, part in parts.items():
        if isinstance(part, str):
            # Reference to another template
            template_fn = refs["templates"][part]
            template_data = template_fn(refs)
            result = execute_template(
                template_data["template"],
                template_data["parts"],
                refs
            )
            values[name] = result["question"]
            executed_parts[name] = result["executed_parts"]
        # Direct value generator
        elif callable(part):
            # Execute the callable
            result = part(refs) if "refs" in part.__code__.co_varnames else part()

            # Check if result is a template
            if isinstance(result, dict) and "template" in result and "parts" in result:
                # Handle nested template
                nested_result = execute_template(
                    result["template"],
                    result["parts"],
                    refs
                )
                values[name] = nested_result["question"]
                executed_parts[name] = nested_result["executed_parts"]
            else:
                # Direct value
                values[name] = str(result)
                executed_parts[name] = result
        else:
            # Static value
            values[name] = str(part)
            executed_parts[name] = part

    return {
        "question": template_str.format(**values),
        "executed_parts": executed_parts
    }

@dataclass
class Template:
    """Defines a template for generating questions"""
    template: str
    parts: Dict[str, str | Callable]
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if "type" not in self.metadata:
            self.metadata["type"] = "direct"

    def eval(self, exercise: Any, rng: random.Random) -> Dict[str, Any]:
        """Evaluate all placeholders and process exercise-specific logic"""
        # Build refs dictionary with generators and shared state
        shared_values = {}
        curriculum = exercise.curriculum  # Get curriculum from exercise
        refs = {
            "dataset_rng": rng,
            "templates": curriculum._symbolic["templates"],
            **{
                name: curriculum.attributes[name].get_generator(
                    curriculum.get_attr_level(name), rng)
                for name in curriculum.attributes.keys()
            },
            **{
                name: lambda refs, name=name, gen=gen: shared_values.setdefault(name, gen(refs))
                for name, gen in curriculum._symbolic.get("shared_vars", {}).items()
            },
            **curriculum._symbolic.get("generators", {})
        }

        result = execute_template(self.template, self.parts, refs)

        # Let exercise process the parts if it has the methods
        if hasattr(exercise, '_parse_expression') and hasattr(exercise, '_evaluate_expression'):
            parsed = exercise._parse_expression(result["executed_parts"])
            answer = exercise._evaluate_expression(parsed)
            return {
                "question": result["question"],
                "answer": answer,
                "metadata": {**self.metadata, "executed_parts": parsed}
            }

        # Default return if exercise doesn't handle parsing/evaluation
        return {
            "question": result["question"],
            "metadata": {**self.metadata, "executed_parts": result["executed_parts"]}
        }