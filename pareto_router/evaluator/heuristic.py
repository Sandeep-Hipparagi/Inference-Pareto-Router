"""Ultra-fast regex/token heuristic evaluator for sub-millisecond dual-axis triage."""

import re
from typing import Any
from pareto_router.evaluator.base import BaseEvaluator


class HeuristicEvaluator(BaseEvaluator):
    """Zero-weight feature extractor evaluating logic and knowledge axes in < 0.5ms."""

    # Single-word Logic indicators
    LOGIC_WORDS = {
        "prove", "proof", "theorem", "lemma", "invariant", "induction",
        "algorithm", "complexity", "leetcode", "codeforces", "euler",
        "recursion", "memoization", "backtracking", "regex", "automata", "cfg", "ast",
        "deadlock", "mutex", "semaphore", "idempotent", "acid", "serialize",
        "deserialization", "segfault", "pointer", "bitwise", "xor", "matrix",
        "eigenvalue", "calculus", "derivative", "integral", "differential",
        "boolean", "satisfiability", "iff", "contrapositive", "counterexample",
        "assert", "assertion", "benchmark", "profiling", "asymptotic",
        "byzantine", "consensus", "raft", "paxos", "liveness", "distributed", "lock"
    }

    # Multi-word Logic phrases
    LOGIC_PHRASES = [
        "binary search", "dynamic programming", "time complexity", "space complexity",
        "big-o", "big o", "o(n", "o(1", "o(log", "race condition", "thread-safe",
        "finite automata", "regular expression", "null pointer", "stack overflow",
        "memory leak", "truth table", "time limit exceeded", "strictly satisfy",
        "must satisfy", "test case", "unit test", "edge case", "divide and conquer",
        "depth first", "breadth first", "shortest path", "topological sort",
        "byzantine fault", "fault tolerant", "lock manager", "network partition"
    ]

    CODE_PATTERNS = [
        re.compile(r"```[\s\S]*?```"),                      # Code fences
        re.compile(r"\b(def|class|async|await|return|fn|func|let|const|var|import|from)\b"),
        re.compile(r"\b(public|private|protected|static|void|struct|interface|type)\b"),
        re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|JOIN|GROUP BY|ORDER BY)\b", re.IGNORECASE),
        re.compile(r"[a-zA-Z0-9_]+\([a-zA-Z0-9_,\s]*\)\s*(->|=>|:)"), # Function signatures
        re.compile(r"(\+\+|--|&&|\|\||==|!=|<=|>=|=>|->|\+=|-=|\*=|/=|%=)"), # Operators
        re.compile(r"\b(for|while|if|else|elif|switch|case|break|continue)\b\s*[\(\{]"),
        re.compile(r"\b(rust|python|golang|typescript|c\+\+|java|sql)\b", re.IGNORECASE),
    ]

    MATH_PATTERNS = [
        re.compile(r"(\$|\\\[|\\\(|\\sum|\\int|\\frac|\\sqrt|\\alpha|\\beta|\\gamma)"),
        re.compile(r"\b\d+\s*[\+\-\*\/\^%]\s*\d+\b"),
        re.compile(r"\b(solve for|find x|equation|polynomial|matrix|modulus)\b", re.IGNORECASE),
    ]

    # Single-word Knowledge indicators
    KNOWLEDGE_WORDS = {
        "history", "historical", "century", "dynasty", "treaty", "revolution",
        "philosophy", "philosophical", "epistemology", "metaphysics", "existentialism",
        "literature", "metaphor", "allegory", "dramaturgy", "symbolism", "sonnet",
        "jurisdiction", "constitutional", "statute", "precedent", "tort", "due process",
        "biology", "pathology", "pharmacology", "receptor", "enzyme", "transcription",
        "geopolitics", "macroeconomics", "monetary policy", "inflation", "fiscal",
        "sociology", "anthropology", "linguistics", "etymology", "phonology"
    }

    # Multi-word Knowledge phrases
    KNOWLEDGE_PHRASES = [
        "trade-offs", "tradeoffs", "pros and cons", "compare and contrast",
        "historical context", "cultural nuance", "geopolitical implications",
        "socioeconomic impact", "ethical implications", "evolution of"
    ]

    # Low-entropy & Aesthetic triggers
    GREETING_WORDS = {
        "hi", "hello", "hey", "howdy", "sup", "greetings", "morning", "afternoon", "evening"
    }

    GREETING_PATTERNS = [
        re.compile(r"\b(hello|hi|hey|howdy|good\s+(morning|afternoon|evening|day))\b", re.IGNORECASE),
        re.compile(r"\b(who are you|what is your name|how are you|how's it going|what's up)\b", re.IGNORECASE),
        re.compile(r"\b(thanks|thank you|ok|okay|cool|awesome|great)\b", re.IGNORECASE),
    ]

    AESTHETIC_REWRITE_PATTERNS = [
        re.compile(r"\bmake\b.*?\b(professional|casual|polite|concise|friendly|punchy|better)\b", re.IGNORECASE),
        re.compile(r"\b(fix (the )?(grammar|spelling|typos)|proofread|rephrase|paraphrase|rewrite this)\b", re.IGNORECASE),
        re.compile(r"\b(translate (this |the following )?(to|into) \w+)\b", re.IGNORECASE),
        re.compile(r"\b(format (this|as) (markdown|bullet points|a table|json))\b", re.IGNORECASE),
        re.compile(r"\b(summarize (this|the following) in \d+ (words|bullets|sentences))\b", re.IGNORECASE),
    ]

    def _compute_scores(
        self,
        text: str,
        context: list[dict[str, str]] | None = None
    ) -> tuple[float, float, dict[str, Any]]:
        # Merge context if present
        full_text = text
        if context:
            recent_context = " ".join(turn.get("content", "") for turn in context[-3:])
            full_text = f"{recent_context}\n{text}"

        cleaned = full_text.strip()
        lower_text = cleaned.lower()
        words = re.findall(r"\b\w+\b", lower_text)
        word_set = set(words)
        num_words = max(1, len(words))

        # Check for greeting & low-entropy chitchat
        has_greeting_word = any(gw in word_set for gw in self.GREETING_WORDS)
        has_greeting_phrase = any(pat.search(cleaned) for pat in self.GREETING_PATTERNS)
        is_greeting = (has_greeting_word or has_greeting_phrase) and num_words <= 12

        # Check for aesthetic rewrite
        is_aesthetic = any(pat.search(cleaned) for pat in self.AESTHETIC_REWRITE_PATTERNS)
        is_low_entropy = is_greeting or is_aesthetic

        # --- Compute Logic Score ---
        logic_points = 0.0

        # Keyword matches
        logic_kw_matches = list(word_set.intersection(self.LOGIC_WORDS))
        logic_points += len(logic_kw_matches) * 0.25

        # Phrase matches
        matched_phrases = [phrase for phrase in self.LOGIC_PHRASES if phrase in lower_text]
        logic_points += len(matched_phrases) * 0.35

        # Code syntax matches
        for pat in self.CODE_PATTERNS:
            matches = pat.findall(full_text)
            if matches:
                logic_points += min(0.35, len(matches) * 0.12)

        # Math matches
        for pat in self.MATH_PATTERNS:
            matches = pat.findall(full_text)
            if matches:
                logic_points += min(0.30, len(matches) * 0.15)

        # Constraint & verifiability phrasing
        if re.search(r"\b(strictly|must satisfy|output only|test case|assertion|unit test|falsify|correctness)\b", lower_text):
            logic_points += 0.20

        # Compute continuous logic score
        logic_score = min(1.0, logic_points)

        # If it's a pure greeting or short aesthetic rewrite without code/math, logic is negligible
        if is_low_entropy and logic_points < 0.25:
            logic_score = 0.02

        # --- Compute Knowledge / Nuance Score ---
        knowledge_points = 0.15  # Baseline

        # Keyword matches
        knowledge_kw_matches = list(word_set.intersection(self.KNOWLEDGE_WORDS))
        knowledge_points += len(knowledge_kw_matches) * 0.20

        # Phrase matches
        matched_k_phrases = [phrase for phrase in self.KNOWLEDGE_PHRASES if phrase in lower_text]
        knowledge_points += len(matched_k_phrases) * 0.25

        # Vocabulary sophistication: word length and rarity proxy
        long_words = [w for w in words if len(w) >= 8]
        long_word_ratio = len(long_words) / num_words
        knowledge_points += min(0.35, long_word_ratio * 1.5)

        # Capitalized entity density (excluding sentence starters)
        caps = re.findall(r"(?<!^)(?<!\.\s)[A-Z][a-z]{2,}", text)
        if len(caps) >= 3:
            knowledge_points += min(0.25, len(caps) * 0.05)

        # Adjust for length
        if num_words > 120:
            knowledge_points += 0.15
        elif num_words > 50:
            knowledge_points += 0.08
        elif num_words < 10 and not is_low_entropy:
            knowledge_points = max(0.1, knowledge_points * 0.6)

        if is_greeting:
            knowledge_score = 0.05
        else:
            knowledge_score = min(1.0, max(0.05, knowledge_points))

        all_matched_logic = logic_kw_matches + matched_phrases
        all_matched_knowledge = knowledge_kw_matches + matched_k_phrases

        metadata: dict[str, Any] = {
            "is_low_entropy": is_low_entropy,
            "is_greeting": is_greeting,
            "is_aesthetic": is_aesthetic,
            "matched_logic_keywords": all_matched_logic[:5],
            "matched_knowledge_keywords": all_matched_knowledge[:5],
            "word_count": num_words,
        }

        return round(logic_score, 3), round(knowledge_score, 3), metadata
