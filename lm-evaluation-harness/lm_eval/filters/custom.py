from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter

import re


@register_filter("custom")
class CustomFilter(Filter):
    """
    Custom filter that applies a custom, user-defined function to the model responses.
    """

    def __init__(self, **kwargs) -> None:
        self.filter_fn = kwargs.pop("filter_fn")

        super().__init__(**kwargs)

    def apply(self, resps, docs):
        return self.filter_fn(resps, docs)


@register_filter("r1_mcqa")
class RegexFilter(Filter):
    """A filter that extracts values from text using regex pattern matching.

    This filter applies a regex pattern to each model response and extracts matched values.
    If no match is found, returns a fallback value. Useful for extracting structured data
    (like numbers) from unstructured model outputs.
    """

    def __init__(
        self,
        regex_pattern_boxed: str = r"\\boxed\{\s*(.*?)\s*\}",
        regex_pattern_answer: str = r"(?i)^(?:[\s\S]*Answer:\s*)([\s\S]*)$",
        group_select: int = 0,
        fallback: str = "[invalid]",
    ) -> None:
        """
        pass a string `regex` to run `re.compile(r"regex")` on.
        `fallback` defines the output returned if no matches for the regex are located.
        """
        self.regex_pattern_boxed = regex_pattern_boxed
        self.regex_boxed = re.compile(regex_pattern_boxed)

        self.regex_pattern_answer = regex_pattern_answer
        self.regex_answer = re.compile(regex_pattern_answer)

        self.group_select = group_select
        self.fallback = fallback

    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)
        def filter_set(inst):
            filtered = []
            for resp in inst:
                match = self.regex_boxed.findall(resp)
                if match:
                    match = match[self.group_select]
                    if isinstance(match, tuple):
                        match = [m for m in match if m]
                        if match:
                            match = match[0]
                        else:
                            match = self.fallback
                    match = match.strip()
                else:
                    match = self.regex_answer.findall(resp)
                    if match:
                        match = match[self.group_select]
                        if isinstance(match, tuple):
                            match = [m for m in match if m]
                            if match:
                                match = match[0]
                            else:
                                match = self.fallback
                        match = match.strip()
                        match = match.split()[0]
                    else:
                        match = self.fallback
                filtered.append(match)
            return filtered

        filtered_resps = list(map(lambda x: filter_set(x), resps))
        filtered_resps = map(lambda r: r[0], filtered_resps) # select first
        return filtered_resps