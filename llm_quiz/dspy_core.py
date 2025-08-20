"""
Simplified LLM Quiz Challenge using DSPy structured output.

This module replaces the complex manual prompt engineering and JSON parsing
with clean DSPy signatures and modules.
"""

import json
import logging
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import dspy
from tqdm import tqdm

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for Python < 3.11

try:
    from .dspy_signatures import (
        AnswerQuizQuestion,
        EvaluateAnswer,
        GenerateFeedback,
        GenerateRevisionGuidance,
        ParseQuestionAndAnswer,
        ValidateQuestion,
        ValidateQuestionSimilarity,
        ValidationIssue,
    )
except ImportError:
    # Handle relative import for standalone execution
    from dspy_signatures import (
        AnswerQuizQuestion,
        EvaluateAnswer,
        GenerateFeedback,
        GenerateRevisionGuidance,
        ParseQuestionAndAnswer,
        ValidateQuestion,
        ValidateQuestionSimilarity,
        ValidationIssue,
    )

logger = logging.getLogger(__name__)


@dataclass
class QuizQuestion:
    """Represents a single quiz question created by the student."""

    question: str  # Student's question
    answer: str  # Student's provided correct answer
    number: int


@dataclass
class RevisionGuidance:
    """Detailed revision guidance for a question."""

    revision_priority: str
    specific_issues: List[str]
    concrete_suggestions: List[str]
    example_improvements: List[str]
    difficulty_adjustment: str
    context_alignment: str
    clarity_improvements: List[str]


@dataclass
class QuizResult:
    """Result for a single question created by the student."""

    question: QuizQuestion  # Student's question and correct answer
    llm_answer: str  # LLM's attempt at answering the student's question
    is_valid: bool  # Whether the student's question is valid
    student_wins: bool  # True if student wins (LLM got it wrong)
    evaluation_explanation: str  # Explanation of how LLM's answer was evaluated
    validation_issues: List[str]  # Issues found with student's question
    revision_guidance: Optional[RevisionGuidance] = (
        None  # Guidance for improving student's question
    )
    difficulty_assessment: Optional[str] = None  # Assessment of question difficulty
    improvement_suggestions: List[str] = None  # Suggestions for improving student's question
    clarity_score: Optional[str] = None  # Assessment of question clarity
    error: Optional[str] = None


@dataclass
class QuizResults:
    """Complete quiz challenge results."""

    quiz_title: str
    total_questions: int
    valid_questions: int
    invalid_questions: int
    student_wins: int
    llm_wins: int
    success_rate: float
    question_results: List[QuizResult]
    feedback_summary: str
    student_passes: bool
    github_classroom_result: str
    similarity_analysis: Optional[Dict[str, Any]] = None  # Added similarity analysis results


class DSPyQuizChallenge:
    """Simplified LLM Quiz Challenge using DSPy structured output."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        quiz_model: str,
        evaluator_model: str,
        context_urls_file: Optional[str] = None,
    ):
        """Initialize the DSPy-based quiz challenge system."""

        # Configure DSPy with the provided LLM
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.quiz_model = quiz_model
        self.evaluator_model = evaluator_model

        # Set up DSPy LM - we'll create a simple wrapper for the existing API
        self.lm = self._create_dspy_lm()
        # Don't configure globally - use dspy.context() in async environments
        
        # Load context content if provided
        self.context_content = (
            self._load_context_from_urls_file(context_urls_file) if context_urls_file else None
        )

        # Initialize DSPy predictors - they will use the context when called
        self.question_parser = dspy.Predict(ParseQuestionAndAnswer)
        self.question_validator = dspy.ChainOfThought(ValidateQuestion)
        self.similarity_validator = dspy.ChainOfThought(ValidateQuestionSimilarity)
        self.question_answerer = dspy.ChainOfThought(AnswerQuizQuestion)
        self.answer_evaluator = dspy.ChainOfThought(EvaluateAnswer)
        self.feedback_generator = dspy.ChainOfThought(GenerateFeedback)
        self.revision_guide_generator = dspy.ChainOfThought(GenerateRevisionGuidance)

        logger.info(
            f"DSPy Quiz Challenge initialized with models: quiz={quiz_model}, evaluator={evaluator_model}"
        )

    def _create_dspy_lm(self):
        """Create a DSPy LM wrapper for the existing API endpoint."""
        logger.debug(
            f"Creating DSPy LM with base_url: {self.base_url}, evaluator_model: {self.evaluator_model}"
        )

        try:
            if "openrouter" in self.base_url.lower():
                # OpenRouter
                lm = dspy.LM(
                    model=self.evaluator_model, api_base=self.base_url, api_key=self.api_key
                )
                logger.debug("Created OpenRouter DSPy LM")
                return lm
            elif "ollama" in self.base_url.lower() or ":11434" in self.base_url.lower():
                # Ollama
                lm = dspy.LM(
                    model=self.evaluator_model, api_base=self.base_url, api_key=self.api_key
                )
                logger.debug("Created Ollama DSPy LM")
                return lm
            else:
                # Default OpenAI-compatible
                lm = dspy.LM(
                    model=self.evaluator_model, api_base=self.base_url, api_key=self.api_key
                )
                logger.debug("Created OpenAI-compatible DSPy LM")
                return lm
        except Exception as e:
            logger.error(f"Error creating DSPy LM: {e}")
            raise

    def _load_context_from_urls_file(self, urls_file: str) -> Optional[str]:
        """Load context content from URLs file."""
        try:
            with open(urls_file, "r") as f:
                urls = [
                    line.strip() for line in f if line.strip() and not line.strip().startswith("#")
                ]

            if not urls:
                logger.warning(f"No URLs found in {urls_file}")
                return None

            combined_content = []
            print(f"Loading {len(urls)} context URL(s)...")

            for i, url in enumerate(urls, 1):
                try:
                    print(f"  Fetching {i}/{len(urls)}: {url}")
                    req = urllib.request.Request(url, headers={"User-Agent": "llm-quiz-challenge"})
                    with urllib.request.urlopen(req, timeout=30) as response:
                        content = response.read().decode("utf-8")
                        filename = url.split("/")[-1] if "/" in url else f"content_{i}"
                        combined_content.append(f"# {filename} (from {url})\n\n{content}")
                        logger.info(f"Loaded content from {url}")
                        print(f"  ✓ Loaded {len(content)} characters")
                except Exception as e:
                    logger.error(f"Error loading {url}: {e}")
                    print(f"  ✗ Failed to load: {e}")

            if combined_content:
                return "\n\n" + "=" * 80 + "\n\n".join(combined_content)

        except Exception as e:
            logger.error(f"Error loading context from {urls_file}: {e}")

        return None

    def _extract_context_topics(self) -> List[str]:
        """Extract main topics from context content for better revision guidance."""
        # Simplified - let the LLM figure out the topics from the context
        return []

    def _validate_question_similarity(self, questions: List[QuizQuestion]) -> Dict[str, Any]:
        """Validate questions for similarity and overlap."""
        if len(questions) <= 1:
            return {
                'has_issues': False,
                'duplicate_pairs': [],
                'overlap_pairs': [],
                'similarity_details': [],
                'overall_assessment': 'Only one question provided, no similarity issues.'
            }

        try:
            logger.debug(f"Checking similarity for {len(questions)} questions...")
            similarity_result = self.similarity_validator(
                questions=[q.question for q in questions],
                answers=[q.answer for q in questions]
            )
            
            # Check if similarity_result is None or missing required attributes
            if similarity_result is None:
                raise ValueError("Similarity validator returned None")
            
            # Safely access attributes with defaults
            has_duplicates = getattr(similarity_result, 'has_duplicates', False)
            has_overlaps = getattr(similarity_result, 'has_overlaps', False)
            duplicate_pairs = getattr(similarity_result, 'duplicate_pairs', [])
            overlap_pairs = getattr(similarity_result, 'overlap_pairs', [])
            similarity_details = getattr(similarity_result, 'similarity_details', [])
            overall_assessment = getattr(similarity_result, 'overall_assessment', 'Unable to assess similarity')

            return {
                'has_issues': has_duplicates or has_overlaps,
                'duplicate_pairs': duplicate_pairs,
                'overlap_pairs': overlap_pairs,
                'similarity_details': similarity_details,
                'overall_assessment': overall_assessment
            }
        except Exception as e:
            logger.error(f"Error validating question similarity: {e}")
            return {
                'has_issues': False,
                'duplicate_pairs': [],
                'overlap_pairs': [],
                'similarity_details': [f"Error checking similarity: {str(e)}"],
                'overall_assessment': 'Unable to check question similarity due to system error.'
            }

    def _apply_similarity_issues_to_questions(
        self, question_results: List[QuizResult], similarity_analysis: Dict[str, Any]
    ) -> None:
        """Apply similarity issues to individual question results."""
        if similarity_analysis is None:
            logger.warning("Similarity analysis is None, skipping issue application")
            return
        
        if not similarity_analysis.get('has_issues', False):
            return

        # Parse duplicate pairs and add issues to affected questions
        for pair_str in similarity_analysis.get('duplicate_pairs', []):
            try:
                if '-' in pair_str:
                    idx1, idx2 = map(int, pair_str.split('-'))
                    # Convert to 0-based indexing if needed
                    if idx1 > 0 and idx1 <= len(question_results):
                        idx1 -= 1
                    if idx2 > 0 and idx2 <= len(question_results):
                        idx2 -= 1
                    
                    for idx in [idx1, idx2]:
                        if 0 <= idx < len(question_results):
                            if ValidationIssue.DUPLICATE_QUESTION.value not in question_results[idx].validation_issues:
                                question_results[idx].validation_issues.append(ValidationIssue.DUPLICATE_QUESTION.value)
            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing duplicate pair '{pair_str}': {e}")

        # Parse overlap pairs and add issues to affected questions  
        for pair_str in similarity_analysis.get('overlap_pairs', []):
            try:
                if '-' in pair_str:
                    idx1, idx2 = map(int, pair_str.split('-'))
                    # Convert to 0-based indexing if needed
                    if idx1 > 0 and idx1 <= len(question_results):
                        idx1 -= 1
                    if idx2 > 0 and idx2 <= len(question_results):
                        idx2 -= 1
                    
                    for idx in [idx1, idx2]:
                        if 0 <= idx < len(question_results):
                            if ValidationIssue.OVERLAPPING_CONTENT.value not in question_results[idx].validation_issues:
                                question_results[idx].validation_issues.append(ValidationIssue.OVERLAPPING_CONTENT.value)
            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing overlap pair '{pair_str}': {e}")

    def load_quiz_from_file(self, quiz_file: Path) -> List[QuizQuestion]:
        """Load quiz from TOML file."""
        try:
            with open(quiz_file, "rb") as f:
                quiz_data = tomllib.load(f)

            questions = []
            for i, q_data in enumerate(quiz_data.get("questions", []), 1):
                questions.append(
                    QuizQuestion(
                        question=q_data.get("question", ""),
                        answer=q_data.get("answer", ""),
                        number=i,
                    )
                )

            logger.info(f"Loaded {len(questions)} questions from {quiz_file}")
            return questions

        except Exception as e:
            logger.error(f"Error loading quiz file {quiz_file}: {e}")
            raise

    def parse_raw_input(self, raw_input: str) -> List[QuizQuestion]:
        """Parse quiz questions from raw student input."""
        try:
            # Use DSPy to parse the input - much simpler than manual parsing!
            result = self.question_parser(raw_input=raw_input)

            questions = []
            for i, (q, a, has_a) in enumerate(
                zip(result.questions, result.answers, result.has_answers), 1
            ):
                questions.append(QuizQuestion(question=q, answer=a if has_a else "", number=i))

            logger.info(f"Parsed {len(questions)} questions from raw input")
            return questions

        except Exception as e:
            logger.error(f"Error parsing raw input: {e}")
            raise

    def _generate_revision_guidance(
        self,
        question: QuizQuestion,
        validation_result: Any,
        llm_response: Optional[str] = None,
        evaluation_result: Optional[Any] = None,
    ) -> RevisionGuidance:
        """Generate detailed revision guidance for a student's question."""
        try:
            context_topics = self._extract_context_topics()

            # Add extra error handling around the DSPy predictor call
            try:
                guidance = self.revision_guide_generator(
                    question=question.question,
                    answer=question.answer,
                    validation_issues=(
                        [issue.value if hasattr(issue, 'value') else str(issue) for issue in validation_result.issues]
                        if validation_result and hasattr(validation_result, "issues")
                        else []
                    ),
                    llm_response=llm_response,
                    evaluation_result=getattr(evaluation_result, 'explanation', None) if evaluation_result else None,
                    context_topics=context_topics,
                )
            except Exception as e:
                logger.error(f"Error in revision_guide_generator DSPy call: {e}")
                guidance = None
            
            # Check if guidance is None
            if guidance is None:
                logger.warning("Revision guide generator returned None, creating fallback guidance")
                guidance = type('MockGuidance', (), {
                    'revision_priority': 'MEDIUM',
                    'specific_issues': ['Unable to analyze question details due to technical error'],
                    'concrete_suggestions': ['Review the question against course materials'],
                    'example_improvements': ['Make the question more specific and clear']
                })()

            return RevisionGuidance(
                revision_priority=getattr(guidance, 'revision_priority', 'MEDIUM'),
                specific_issues=getattr(guidance, 'specific_issues', ['Unable to analyze question details']),
                concrete_suggestions=getattr(guidance, 'concrete_suggestions', ['Review the question against course materials']),
                example_improvements=getattr(guidance, 'example_improvements', ['Make the question more specific']),
                difficulty_adjustment=getattr(guidance, 'difficulty_adjustment', 'Review question complexity'),
                context_alignment=getattr(guidance, 'context_alignment', 'Align with course topics'),
                clarity_improvements=getattr(guidance, 'clarity_improvements', ['Make the question clearer']),
            )
        except Exception as e:
            logger.error(f"Error generating revision guidance: {e}")
            # Provide basic fallback guidance
            return RevisionGuidance(
                revision_priority="MEDIUM",
                specific_issues=["Unable to generate detailed analysis"],
                concrete_suggestions=["Review the question against the provided context materials"],
                example_improvements=["Make the question more specific to the course topics"],
                difficulty_adjustment="Ensure the question requires deep understanding rather than memorization",
                context_alignment="Align the question with the provided context materials",
                clarity_improvements=["Make the question more precise and specific", "Avoid ambiguous wording"],
            )

    def run_quiz_challenge(
        self, questions: List[QuizQuestion], quiz_title: str = "Quiz Challenge"
    ) -> QuizResults:
        """Run the complete quiz challenge using DSPy structured output."""
        logger.info(f"Starting DSPy quiz challenge with {len(questions)} questions")
        
        # Step 1: Validate question similarity first
        logger.info("Validating question similarity and overlap...")
        # Skip similarity analysis for single questions to avoid NoneType errors
        similarity_analysis = None
        if len(questions) > 1:
            similarity_analysis = self._validate_question_similarity(questions)
        else:
            logger.info("Only one question provided, skipping similarity analysis")
        if similarity_analysis and similarity_analysis.get('has_issues', False):
            print(f"\n🔍 SIMILARITY ANALYSIS RESULTS:")
            duplicate_count = len(similarity_analysis.get('duplicate_pairs', []))
            overlap_count = len(similarity_analysis.get('overlap_pairs', []))
            print(f"   Found {duplicate_count} duplicate pairs and {overlap_count} overlap pairs")
            
            # Show the actual questions for reference
            print(f"   📚 Your questions:")
            for i, q in enumerate(questions, 1):
                print(f"      Q{i}: {q.question}")
            
            if similarity_analysis['duplicate_pairs']:
                print(f"   📋 Duplicate pairs: {', '.join(similarity_analysis['duplicate_pairs'])}")
            if similarity_analysis['overlap_pairs']:
                print(f"   🔄 Overlapping pairs: {', '.join(similarity_analysis['overlap_pairs'])}")
                
            print(f"   📝 Detailed analysis:")
            for i, detail in enumerate(similarity_analysis['similarity_details'], 1):
                print(f"      {i}. {detail}")
            
            print(f"   🎯 Overall assessment: {similarity_analysis['overall_assessment']}")
            print(f"   ℹ️  Note: Similarity issues are informational and don't affect pass/fail if you legitimately won all questions\n")
            
            logger.warning(f"Similarity issues found: {len(similarity_analysis['duplicate_pairs'])} duplicate pairs, {len(similarity_analysis['overlap_pairs'])} overlap pairs")
        else:
            print(f"✅ No similarity issues found between questions\n")

        question_results = []
        valid_count = 0
        student_wins = 0
        llm_wins = 0
        all_validation_issues = []

        # Create progress bar with more granular steps
        # Each question has: validate -> generate guidance -> LLM answer -> evaluate -> finalize
        # Add 1 extra step for similarity validation
        total_steps = len(questions) * 5 + 1
        pbar = tqdm(total=total_steps, desc="Processing quiz", unit="step")
        
        # Mark similarity validation as complete
        pbar.set_description("Similarity validation complete")
        pbar.update(1)

        for question in questions:
            logger.info(f"Processing question {question.number}: {question.question[:50]}...")

            try:
                # Step 1: Validate the student's question using DSPy
                pbar.set_description(f"Q{question.number}: Validating question")
                logger.debug(f"Validating student's question {question.number} with DSPy...")
                validation = self.question_validator(
                    question=question.question,
                    answer=question.answer,
                    context_content=self.context_content,
                )
                
                # Check if validation result is None
                if validation is None:
                    raise ValueError("Question validator returned None")
                    
                logger.debug(
                    f"Validation result: valid={getattr(validation, 'is_valid', False)}, reason={getattr(validation, 'reason', 'Unknown')}"
                )
                pbar.update(1)  # Step 1 complete

                # Step 2: Generate revision guidance for all questions (valid or invalid)
                pbar.set_description(f"Q{question.number}: Generating guidance")
                try:
                    revision_guidance = self._generate_revision_guidance(question, validation)
                except Exception as e:
                    logger.error(f"Error generating revision guidance for question {question.number}: {e}")
                    # Create a fallback revision guidance
                    revision_guidance = RevisionGuidance(
                        revision_priority='MEDIUM',
                        specific_issues=['Unable to analyze question due to technical error'],
                        concrete_suggestions=['Review the question against course materials'],
                        example_improvements=['Make the question more specific and clear'],
                        context_alignment='Unknown'
                    )
                pbar.update(1)  # Step 2 complete

                is_valid = getattr(validation, 'is_valid', False)
                if not is_valid:
                    pbar.set_description(f"Q{question.number}: Question invalid, skipping")
                    reason = getattr(validation, 'reason', 'Unknown validation error')
                    logger.warning(
                        f"Student's question {question.number} failed validation: {reason}"
                    )
                    issues = getattr(validation, 'issues', [])
                    all_validation_issues.extend([issue.value if hasattr(issue, 'value') else str(issue) for issue in issues])

                    result = QuizResult(
                        question=question,
                        llm_answer="Question rejected during validation",
                        is_valid=False,
                        student_wins=False,
                        evaluation_explanation=f"Invalid student question: {reason}",
                        validation_issues=[issue.value if hasattr(issue, 'value') else str(issue) for issue in issues],
                        revision_guidance=revision_guidance,
                        difficulty_assessment=getattr(validation, 'difficulty_assessment', 'APPROPRIATE'),
                        improvement_suggestions=getattr(validation, 'revision_suggestions', []),
                        clarity_score=getattr(validation, 'clarity_score', None),
                        error=reason,
                    )
                    question_results.append(result)
                    # Skip remaining 3 steps for invalid questions
                    pbar.update(3)
                    continue

                valid_count += 1

                # Step 3: LLM attempts to answer the student's question using DSPy
                # We need to switch to the quiz model for this step
                pbar.set_description(f"Q{question.number}: LLM taking quiz")
                logger.debug(
                    f"LLM attempting to answer student's question {question.number} using {self.quiz_model}..."
                )
                with dspy.context(
                    lm=dspy.LM(model=self.quiz_model, api_base=self.base_url, api_key=self.api_key)
                ):
                    llm_response = self.question_answerer(
                        question=question.question, context_content=self.context_content
                    )
                
                # Check if llm_response is None
                if llm_response is None:
                    raise ValueError("Question answerer returned None")
                
                llm_answer = getattr(llm_response, 'answer', 'Unable to generate answer')
                logger.debug(f"LLM's answer: {llm_answer[:100]}...")
                pbar.update(1)  # Step 3 complete

                # Step 4: Evaluate LLM's answer against student's correct answer using DSPy
                pbar.set_description(f"Q{question.number}: Evaluating LLM answer")
                logger.debug(f"Evaluating LLM's answer for question {question.number}...")
                evaluation = self.answer_evaluator(
                    question=question.question,
                    correct_answer=question.answer,
                    llm_answer=llm_answer,
                )
                
                # Check if evaluation is None
                if evaluation is None:
                    raise ValueError("Answer evaluator returned None")
                    
                verdict = getattr(evaluation, 'verdict', 'INCORRECT')
                student_won_this_question = getattr(evaluation, 'student_wins', False)
                logger.debug(
                    f"Evaluation: verdict={verdict}, student_wins={student_won_this_question}"
                )
                pbar.update(1)  # Step 4 complete

                # Step 5: Finalize results
                if student_won_this_question:
                    student_wins += 1
                    pbar.set_description(f"Q{question.number}: Complete - Student wins!")
                else:
                    llm_wins += 1
                    pbar.set_description(f"Q{question.number}: Complete - LLM wins")

                # Generate revision guidance for valid questions too
                revision_guidance = self._generate_revision_guidance(
                    question, validation, llm_answer, evaluation
                )

                result = QuizResult(
                    question=question,
                    llm_answer=llm_answer,
                    is_valid=True,
                    student_wins=student_won_this_question,
                    evaluation_explanation=getattr(evaluation, 'explanation', 'No explanation available'),
                    validation_issues=[],
                    revision_guidance=revision_guidance,
                    difficulty_assessment=getattr(validation, 'difficulty_assessment', 'APPROPRIATE'),
                    improvement_suggestions=getattr(evaluation, 'improvement_suggestions', []),
                    clarity_score=getattr(validation, 'clarity_score', None),
                )
                question_results.append(result)
                pbar.update(1)  # Step 5 complete

            except Exception as e:
                pbar.set_description(f"Q{question.number}: Error occurred")
                logger.error(f"Error processing question {question.number}: {e}")
                logger.debug(f"Full exception details:", exc_info=True)
                result = QuizResult(
                    question=question,
                    llm_answer="System error",
                    is_valid=False,
                    student_wins=False,
                    evaluation_explanation=f"System error: {str(e)}",
                    validation_issues=[],
                    error=str(e),
                )
                question_results.append(result)
                # Update remaining steps for error case
                pbar.update(5 - (pbar.n % 5) if pbar.n % 5 != 0 else 0)

        # Close the progress bar
        pbar.close()

        # Apply similarity issues to individual questions after all processing is complete
        logger.info("Applying similarity issues to question results...")
        if similarity_analysis is not None:
            self._apply_similarity_issues_to_questions(question_results, similarity_analysis)
        
        # Add similarity issues to the overall validation issues list
        if similarity_analysis is not None and similarity_analysis.get('has_issues', False):
            if similarity_analysis.get('duplicate_pairs', []):
                all_validation_issues.extend([ValidationIssue.DUPLICATE_QUESTION.value] * len(similarity_analysis['duplicate_pairs']))
            if similarity_analysis.get('overlap_pairs', []):
                all_validation_issues.extend([ValidationIssue.OVERLAPPING_CONTENT.value] * len(similarity_analysis['overlap_pairs']))

        # Calculate results
        evaluated_questions = student_wins + llm_wins
        success_rate = student_wins / evaluated_questions if evaluated_questions > 0 else 0.0
        
        # Student passes if they win all valid questions - similarity issues are informational only
        student_passes = (
            valid_count == len(questions) and 
            evaluated_questions > 0 and 
            success_rate >= 1.0
            # Removed similarity check - it shouldn't block passing if student legitimately won
        )

        # Generate feedback using DSPy
        try:
            logger.debug("Generating feedback with DSPy...")
            feedback = self.feedback_generator(
                total_questions=len(questions),
                valid_questions=valid_count,
                invalid_questions=len(questions) - valid_count,
                student_wins=student_wins,
                llm_wins=llm_wins,
                validation_issues=all_validation_issues,
                success_rate=success_rate,
            )
            
            # Check if feedback is None
            if feedback is None:
                raise ValueError("Feedback generator returned None")
            
            feedback_summary = getattr(feedback, 'feedback_summary', 'Unable to generate feedback')
            github_result = getattr(feedback, 'github_classroom_marker', 'STUDENTS_QUIZ_KEIKO_LOSE')
            logger.debug(f"DSPy feedback generated successfully: {feedback_summary[:100]}...")
        except Exception as e:
            logger.error(f"Error generating feedback: {e}")
            logger.debug(f"Feedback generation exception details:", exc_info=True)
            # Create manual feedback as fallback
            if evaluated_questions == 0:
                feedback_summary = "No questions were successfully processed. Please check your quiz format and try again."
            elif student_wins == 0:
                feedback_summary = "You did not answer any questions correctly. It appears you may need to review the material more thoroughly. Keep practicing and don't get discouraged!"
            else:
                feedback_summary = f"Quiz completed! You successfully stumped the LLM on {student_wins} out of {evaluated_questions} evaluated questions (success rate: {success_rate:.1%})."
            github_result = (
                "STUDENTS_QUIZ_KEIKO_WIN" if student_passes else "STUDENTS_QUIZ_KEIKO_LOSE"
            )

        return QuizResults(
            quiz_title=quiz_title,
            total_questions=len(questions),
            valid_questions=valid_count,
            invalid_questions=len(questions) - valid_count,
            student_wins=student_wins,
            llm_wins=llm_wins,
            success_rate=success_rate,
            question_results=question_results,
            feedback_summary=feedback_summary,
            student_passes=student_passes,
            github_classroom_result=github_result,
            similarity_analysis=similarity_analysis,  # Added similarity analysis
        )

    def save_results(self, results: QuizResults, output_file: Path):
        """Save quiz results to JSON file."""
        try:
            results_dict = asdict(results)
            with open(output_file, "w") as f:
                json.dump(results_dict, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
