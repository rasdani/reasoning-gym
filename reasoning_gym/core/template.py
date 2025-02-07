"""Core template system for generating questions."""

from typing import Dict, Any, Callable
from dataclasses import dataclass
import random

def execute_template(name: str, template_data: Dict[str, Any], refs: Dict[str, Callable]) -> Dict[str, Any]:
    """Execute a template and return its result.

    Args:
        name: Name of the template being executed
        template_data: Template definition data
        refs: Reference functions and generators

    Returns:
        Dict containing the executed template question and metadata
    """
    # Handle callable template data
    if callable(template_data):
        template_data = template_data(refs)

    template_str = template_data["template"]
    parts = template_data["parts"]
    executed_parts = {}

    # Execute each part of the template
    for part_name, part in parts.items():
        if isinstance(part, str):
            # Handle nested template reference
            nested_result = execute_template(
                part,
                refs["templates"][part](refs),
                refs
            )
            executed_parts[part_name] = nested_result["question"]
        elif callable(part):
            # Handle direct value generator
            value = part(refs) if "refs" in part.__code__.co_varnames else part()
            executed_parts[part_name] = str(value)

    # Format the template with executed parts
    result = {
        "question": template_str.format(**executed_parts),
        "metadata": {
            "template_name": name
        }
    }

    # Only add executed_parts if not already present in metadata
    if "executed_parts" not in result["metadata"]:
        result["metadata"]["executed_parts"] = executed_parts

    return result

@dataclass
class Placeholder:
    """Represents a placeholder in an expression template.

    The placeholder is evaluated using the symbolic template system defined in the curriculum's
    _symbolic["templates"] dictionary. The generator name should match a key in that dictionary.

    Args:
        name: The name of the placeholder to be used in template formatting
        generator: The name of the template in _symbolic["templates"] to use for generation
        args: Optional arguments to pass to the generator (not currently used)
    """
    name: str
    generator: str  # Name of template in _symbolic["templates"] to use
    args: Dict[str, Any] = None

    def eval(self, exercise: Any, rng: random.Random) -> Dict[str, Any]:
        """Evaluate placeholder using curriculum"""
        curriculum = exercise.curriculum

        return execute_template(
            self.generator,  # Use the generator name directly
            curriculum._symbolic["templates"][self.generator],
            {
                "dataset_rng": rng,
                "templates": curriculum._symbolic["templates"],
                **{name: gen for name, gen in curriculum._symbolic.get("generators", {}).items()},
                **{
                    name: lambda attr=name: curriculum.attributes[attr].get_generator(
                        curriculum.get_attr_level(attr), rng)()
                    for name in curriculum.attributes.keys()
                }
            }
        )

@dataclass
class Template:
    """Defines a template for generating questions"""
    question: str
    placeholders: Dict[str, Placeholder]
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if "type" not in self.metadata:
            self.metadata["type"] = "direct"

    def eval(self, exercise: Any, rng: random.Random) -> Dict[str, Any]:
        """Evaluate all placeholders and process exercise-specific logic"""
        values = {}
        metadata = {}

        for name, placeholder in self.placeholders.items():
            result = placeholder.eval(exercise, rng)
            values[name] = result["question"]
            metadata[name] = result["metadata"]

        # Format question text
        question = self.question.format(**values)

        # Let exercise process the parts if it has the methods
        if hasattr(exercise, '_parse_expression') and hasattr(exercise, '_evaluate_expression'):
            parsed = exercise._parse_expression(metadata)
            answer = exercise._evaluate_expression(parsed)
            return {
                "question": question,
                "answer": answer,
                "metadata": parsed
            }

        # Default return if exercise doesn't handle parsing/evaluation
        return {
            "question": question,
            "metadata": metadata
        }