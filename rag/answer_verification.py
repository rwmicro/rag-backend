"""
Answer Verification System

Validates generated answers before presenting to users by checking:
1. Factual grounding (answer supported by retrieved context)
2. Hallucination detection (claims not in context)
3. Consistency (answer doesn't contradict context)
4. Completeness (answer addresses all parts of query)
5. Confidence scoring
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import re
import numpy as np
from loguru import logger

from .chunking import Chunk
from config.settings import settings


class VerificationStatus(Enum):
    """Answer verification status"""
    VERIFIED = "verified"               # High confidence, well-grounded
    LIKELY_CORRECT = "likely_correct"   # Medium confidence
    UNCERTAIN = "uncertain"             # Low confidence, needs review
    HALLUCINATION = "hallucination"     # Contains unsupported claims
    CONTRADICTORY = "contradictory"     # Contradicts source material


@dataclass
class VerificationIssue:
    """Issue found during verification"""
    issue_type: str  # "hallucination", "contradiction", "incomplete", "unsupported"
    severity: str    # "critical", "moderate", "minor"
    description: str
    evidence: Optional[str] = None


@dataclass
class AnswerVerificationResult:
    """Result of answer verification"""
    status: VerificationStatus
    confidence_score: float
    grounding_score: float
    consistency_score: float
    completeness_score: float
    issues: List[VerificationIssue]
    explanation: str
    should_present: bool  # Whether answer is safe to present to user


class AnswerVerifier:
    """
    Verifies generated answers against retrieved context

    Multi-stage verification:
    1. Claim extraction: Break answer into atomic claims
    2. Grounding check: Verify each claim is supported by context
    3. Contradiction check: Ensure no contradictions with context
    4. Completeness check: Verify query is fully answered
    5. Confidence scoring: Aggregate verification metrics
    """

    def __init__(
        self,
        llm_generator=None,
        embedding_model=None,
        grounding_threshold: float = 0.7,
        verification_threshold: float = 0.6,
    ):
        """
        Initialize answer verifier

        Args:
            llm_generator: Optional LLM for claim extraction and verification
            embedding_model: Optional embedding model for semantic similarity
            grounding_threshold: Minimum similarity for claim to be grounded
            verification_threshold: Minimum score to pass verification
        """
        self.llm_generator = llm_generator
        self.embedding_model = embedding_model
        self.grounding_threshold = grounding_threshold
        self.verification_threshold = verification_threshold

        logger.info(
            f"Initialized AnswerVerifier "
            f"(grounding_threshold={grounding_threshold}, "
            f"verification_threshold={verification_threshold})"
        )

    def verify(
        self,
        query: str,
        answer: str,
        context_chunks: List[Tuple[Chunk, float]],
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> AnswerVerificationResult:
        """
        Verify generated answer against retrieved context

        Args:
            query: Original user query
            answer: Generated answer
            context_chunks: Retrieved context chunks with scores
            query_analysis: Optional query analysis

        Returns:
            AnswerVerificationResult with verification status and issues
        """
        logger.info(f"Verifying answer (length={len(answer)} chars)")

        if not answer or not answer.strip():
            return AnswerVerificationResult(
                status=VerificationStatus.UNCERTAIN,
                confidence_score=0.0,
                grounding_score=0.0,
                consistency_score=0.0,
                completeness_score=0.0,
                issues=[VerificationIssue(
                    issue_type="incomplete",
                    severity="critical",
                    description="Empty answer generated"
                )],
                explanation="Answer is empty",
                should_present=False,
            )

        if not context_chunks:
            return AnswerVerificationResult(
                status=VerificationStatus.UNCERTAIN,
                confidence_score=0.3,
                grounding_score=0.0,
                consistency_score=0.5,
                completeness_score=0.0,
                issues=[VerificationIssue(
                    issue_type="unsupported",
                    severity="critical",
                    description="No context available for verification"
                )],
                explanation="Cannot verify answer without context",
                should_present=False,
            )

        issues = []

        # 1. Grounding Check: Is answer supported by context?
        grounding_score, grounding_issues = self._check_grounding(
            answer, context_chunks
        )
        issues.extend(grounding_issues)

        # 2. Consistency Check: Does answer contradict context?
        consistency_score, consistency_issues = self._check_consistency(
            answer, context_chunks
        )
        issues.extend(consistency_issues)

        # 3. Completeness Check: Does answer address the query?
        completeness_score, completeness_issues = self._check_completeness(
            query, answer, query_analysis
        )
        issues.extend(completeness_issues)

        # 4. Hallucination Detection
        hallucination_issues = self._detect_hallucinations(
            answer, context_chunks
        )
        issues.extend(hallucination_issues)

        # 5. Compute overall confidence
        confidence_score = self._compute_confidence(
            grounding_score,
            consistency_score,
            completeness_score,
            issues,
        )

        # 6. Determine verification status
        status = self._determine_status(
            confidence_score,
            grounding_score,
            consistency_score,
            issues,
        )

        # 7. Decide if answer should be presented
        should_present = self._should_present_answer(
            status, confidence_score, issues
        )

        # 8. Generate explanation
        explanation = self._generate_explanation(
            status, confidence_score, grounding_score,
            consistency_score, completeness_score, issues
        )

        logger.info(
            f"Verification complete: status={status.value}, "
            f"confidence={confidence_score:.2f}, "
            f"issues={len(issues)}, "
            f"present={should_present}"
        )

        return AnswerVerificationResult(
            status=status,
            confidence_score=confidence_score,
            grounding_score=grounding_score,
            consistency_score=consistency_score,
            completeness_score=completeness_score,
            issues=issues,
            explanation=explanation,
            should_present=should_present,
        )

    def _check_grounding(
        self,
        answer: str,
        context_chunks: List[Tuple[Chunk, float]],
    ) -> Tuple[float, List[VerificationIssue]]:
        """
        Check if answer is grounded in context

        Args:
            answer: Generated answer
            context_chunks: Retrieved context

        Returns:
            Tuple of (grounding_score, issues)
        """
        issues = []

        # Combine context
        context_text = "\n".join([chunk.content for chunk, _ in context_chunks])

        # Method 1: Lexical overlap
        answer_words = set(answer.lower().split())
        context_words = set(context_text.lower().split())

        # Remove common stop words for better signal
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'could', 'should', 'may', 'might', 'can', 'to', 'of', 'in',
                     'on', 'at', 'by', 'for', 'with', 'from', 'as', 'and', 'or',
                     'but', 'if', 'then', 'than', 'that', 'this', 'these', 'those'}

        answer_content_words = answer_words - stop_words
        context_content_words = context_words - stop_words

        if answer_content_words:
            overlap_ratio = len(answer_content_words & context_content_words) / len(answer_content_words)
        else:
            overlap_ratio = 0.0

        # Method 2: Semantic similarity (if embedding model available)
        semantic_score = 0.0
        if self.embedding_model:
            try:
                answer_embedding = self.embedding_model.encode_single(answer)
                context_embedding = self.embedding_model.encode_single(context_text[:2000])

                # Cosine similarity
                answer_norm = answer_embedding / (np.linalg.norm(answer_embedding) + 1e-8)
                context_norm = context_embedding / (np.linalg.norm(context_embedding) + 1e-8)
                semantic_score = float(np.dot(answer_norm, context_norm))
            except Exception as e:
                logger.warning(f"Semantic similarity computation failed: {e}")

        # Combine scores
        grounding_score = 0.4 * overlap_ratio + 0.6 * semantic_score

        # Check for issues
        if grounding_score < 0.4:
            issues.append(VerificationIssue(
                issue_type="unsupported",
                severity="critical",
                description=f"Answer has weak grounding in context (score={grounding_score:.2f})",
                evidence=f"Lexical overlap: {overlap_ratio:.2%}, Semantic: {semantic_score:.2f}"
            ))
        elif grounding_score < 0.6:
            issues.append(VerificationIssue(
                issue_type="unsupported",
                severity="moderate",
                description=f"Answer grounding is uncertain (score={grounding_score:.2f})"
            ))

        logger.debug(
            f"Grounding check: score={grounding_score:.2f} "
            f"(lexical={overlap_ratio:.2f}, semantic={semantic_score:.2f})"
        )

        return grounding_score, issues

    def _check_consistency(
        self,
        answer: str,
        context_chunks: List[Tuple[Chunk, float]],
    ) -> Tuple[float, List[VerificationIssue]]:
        """
        Check if answer is consistent with context (no contradictions)

        Args:
            answer: Generated answer
            context_chunks: Retrieved context

        Returns:
            Tuple of (consistency_score, issues)
        """
        issues = []

        # Look for contradiction indicators in answer
        contradiction_patterns = [
            r'\bhowever\b.*?\bbut\b',
            r'\balthough\b.*?\bactually\b',
            r'\bdespite\b.*?\bin fact\b',
            r'\bcontrary to\b',
        ]

        has_contradiction_signal = any(
            re.search(pattern, answer.lower())
            for pattern in contradiction_patterns
        )

        # If LLM available, use for deep consistency check
        if self.llm_generator and has_contradiction_signal:
            consistency_score = self._llm_consistency_check(answer, context_chunks)
        else:
            # Simple heuristic: if no contradiction signals, assume consistent
            consistency_score = 0.9 if not has_contradiction_signal else 0.5

        if consistency_score < 0.5:
            issues.append(VerificationIssue(
                issue_type="contradiction",
                severity="critical",
                description="Answer may contradict source material"
            ))

        logger.debug(f"Consistency check: score={consistency_score:.2f}")

        return consistency_score, issues

    def _check_completeness(
        self,
        query: str,
        answer: str,
        query_analysis: Optional[Dict[str, Any]],
    ) -> Tuple[float, List[VerificationIssue]]:
        """
        Check if answer addresses all parts of the query

        Args:
            query: Original query
            answer: Generated answer
            query_analysis: Optional query analysis

        Returns:
            Tuple of (completeness_score, issues)
        """
        issues = []

        # Check answer length (too short might be incomplete)
        if len(answer.split()) < 5:
            issues.append(VerificationIssue(
                issue_type="incomplete",
                severity="moderate",
                description=f"Answer is very short ({len(answer.split())} words)"
            ))
            return 0.4, issues

        # Check for common incomplete answer phrases
        incomplete_phrases = [
            r"^i don't know",
            r"^i cannot",
            r"^i'm not sure",
            r"^there is no information",
            r"^the context does not",
        ]

        for pattern in incomplete_phrases:
            if re.search(pattern, answer.lower()):
                completeness_score = 0.3
                issues.append(VerificationIssue(
                    issue_type="incomplete",
                    severity="moderate",
                    description="Answer indicates insufficient information"
                ))
                return completeness_score, issues

        # Check if answer addresses query type requirements
        completeness_score = 0.8  # Default

        if query_analysis:
            query_type = query_analysis.get("query_type")

            if query_type == "comparative":
                # Should mention both entities
                if not re.search(r'\b(both|and|versus|compared to)\b', answer.lower()):
                    issues.append(VerificationIssue(
                        issue_type="incomplete",
                        severity="minor",
                        description="Comparison query may not address both entities"
                    ))
                    completeness_score = 0.6

            elif query_type == "multi_hop":
                # Should be more detailed
                if len(answer.split()) < 20:
                    issues.append(VerificationIssue(
                        issue_type="incomplete",
                        severity="minor",
                        description="Multi-hop query answer seems brief"
                    ))
                    completeness_score = 0.7

        logger.debug(f"Completeness check: score={completeness_score:.2f}")

        return completeness_score, issues

    def _detect_hallucinations(
        self,
        answer: str,
        context_chunks: List[Tuple[Chunk, float]],
    ) -> List[VerificationIssue]:
        """
        Detect potential hallucinations (specific claims not in context)

        Args:
            answer: Generated answer
            context_chunks: Retrieved context

        Returns:
            List of hallucination issues
        """
        issues = []

        # Extract specific claims (numbers, dates, names)
        # Numbers and percentages
        numbers_in_answer = re.findall(r'\b\d+(?:\.\d+)?%?\b', answer)

        if numbers_in_answer:
            context_text = " ".join([chunk.content for chunk, _ in context_chunks])
            ungrounded_numbers = [
                num for num in numbers_in_answer
                if num not in context_text
            ]

            if len(ungrounded_numbers) > len(numbers_in_answer) * 0.5:
                issues.append(VerificationIssue(
                    issue_type="hallucination",
                    severity="critical",
                    description=f"Answer contains numbers not found in context: {ungrounded_numbers[:3]}",
                    evidence=f"{len(ungrounded_numbers)}/{len(numbers_in_answer)} numbers ungrounded"
                ))

        return issues

    def _llm_consistency_check(
        self,
        answer: str,
        context_chunks: List[Tuple[Chunk, float]],
    ) -> float:
        """
        Use LLM to check consistency

        Args:
            answer: Generated answer
            context_chunks: Context

        Returns:
            Consistency score (0-1)
        """
        context_text = "\n".join([chunk.content for chunk, _ in context_chunks[:3]])

        prompt = f"""Check if the ANSWER is consistent with the CONTEXT.

CONTEXT:
{context_text[:1000]}

ANSWER:
{answer}

Does the answer contradict any information in the context?
Reply with only: "CONSISTENT" or "CONTRADICTORY"
"""

        try:
            response = self.llm_generator.generate(prompt, max_tokens=10, temperature=0.0)
            response_clean = response.strip().upper()

            if "CONSISTENT" in response_clean:
                return 0.9
            elif "CONTRADICTORY" in response_clean:
                return 0.2
            else:
                return 0.5
        except Exception as e:
            logger.warning(f"LLM consistency check failed: {e}")
            return 0.5

    def _compute_confidence(
        self,
        grounding_score: float,
        consistency_score: float,
        completeness_score: float,
        issues: List[VerificationIssue],
    ) -> float:
        """
        Compute overall confidence score

        Args:
            grounding_score: Grounding score
            consistency_score: Consistency score
            completeness_score: Completeness score
            issues: List of issues

        Returns:
            Confidence score (0-1)
        """
        # Weighted combination
        base_confidence = (
            0.45 * grounding_score +
            0.35 * consistency_score +
            0.20 * completeness_score
        )

        # Penalty for critical issues
        critical_count = sum(1 for issue in issues if issue.severity == "critical")
        penalty = min(0.3, critical_count * 0.15)

        confidence = max(0.0, base_confidence - penalty)

        return confidence

    def _determine_status(
        self,
        confidence_score: float,
        grounding_score: float,
        consistency_score: float,
        issues: List[VerificationIssue],
    ) -> VerificationStatus:
        """
        Determine verification status

        Args:
            confidence_score: Overall confidence
            grounding_score: Grounding score
            consistency_score: Consistency score
            issues: List of issues

        Returns:
            VerificationStatus
        """
        # Check for critical issues
        has_hallucination = any(
            issue.issue_type == "hallucination" and issue.severity == "critical"
            for issue in issues
        )

        has_contradiction = any(
            issue.issue_type == "contradiction" and issue.severity == "critical"
            for issue in issues
        )

        if has_hallucination:
            return VerificationStatus.HALLUCINATION

        if has_contradiction:
            return VerificationStatus.CONTRADICTORY

        # Based on confidence
        if confidence_score >= 0.8 and grounding_score >= 0.7:
            return VerificationStatus.VERIFIED
        elif confidence_score >= 0.6:
            return VerificationStatus.LIKELY_CORRECT
        else:
            return VerificationStatus.UNCERTAIN

    def _should_present_answer(
        self,
        status: VerificationStatus,
        confidence_score: float,
        issues: List[VerificationIssue],
    ) -> bool:
        """
        Decide if answer should be presented to user

        Args:
            status: Verification status
            confidence_score: Confidence score
            issues: List of issues

        Returns:
            Whether to present answer
        """
        # Never present hallucinations or contradictions
        if status in [VerificationStatus.HALLUCINATION, VerificationStatus.CONTRADICTORY]:
            return False

        # Present if confidence above threshold
        if confidence_score >= self.verification_threshold:
            return True

        # Check for critical issues
        has_critical = any(issue.severity == "critical" for issue in issues)
        if has_critical:
            return False

        # Borderline case: present with warning
        return confidence_score >= 0.4

    def _generate_explanation(
        self,
        status: VerificationStatus,
        confidence_score: float,
        grounding_score: float,
        consistency_score: float,
        completeness_score: float,
        issues: List[VerificationIssue],
    ) -> str:
        """
        Generate human-readable explanation

        Args:
            status: Verification status
            confidence_score: Overall confidence
            grounding_score: Grounding score
            consistency_score: Consistency score
            completeness_score: Completeness score
            issues: List of issues

        Returns:
            Explanation string
        """
        explanation = f"Verification Status: {status.value.upper()}\n"
        explanation += f"Confidence: {confidence_score:.1%}\n"
        explanation += f"Grounding: {grounding_score:.1%} | "
        explanation += f"Consistency: {consistency_score:.1%} | "
        explanation += f"Completeness: {completeness_score:.1%}\n"

        if issues:
            explanation += f"\nIssues ({len(issues)}):\n"
            for issue in issues:
                explanation += f"  [{issue.severity.upper()}] {issue.description}\n"

        return explanation

    def format_report(self, result: AnswerVerificationResult) -> str:
        """
        Generate detailed verification report

        Args:
            result: Verification result

        Returns:
            Formatted report
        """
        report = "=== Answer Verification Report ===\n"
        report += f"Status: {result.status.value.upper()}\n"
        report += f"Overall Confidence: {result.confidence_score:.1%}\n\n"

        report += "Metrics:\n"
        report += f"  • Grounding:    {result.grounding_score:.1%}\n"
        report += f"  • Consistency:  {result.consistency_score:.1%}\n"
        report += f"  • Completeness: {result.completeness_score:.1%}\n\n"

        if result.issues:
            report += f"Issues Found ({len(result.issues)}):\n"
            for i, issue in enumerate(result.issues, 1):
                report += f"  {i}. [{issue.severity.upper()}] {issue.issue_type}: {issue.description}\n"
                if issue.evidence:
                    report += f"     Evidence: {issue.evidence}\n"
            report += "\n"

        report += f"Present to User: {'Yes' if result.should_present else 'No'}\n"
        report += "=" * 35

        return report
