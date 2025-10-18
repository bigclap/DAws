//! Synthetic evaluation bench covering symbolic reasoning and retrieval tasks.

/// Single evaluation example with an expected textual answer and cosine requirement.
#[derive(Clone, Debug)]
pub struct SyntheticCase {
    query: String,
    expected: String,
    cosine_threshold: f32,
}

impl SyntheticCase {
    /// Creates a new synthetic case.
    pub fn new(
        query: impl Into<String>,
        expected: impl Into<String>,
        cosine_threshold: f32,
    ) -> Self {
        Self {
            query: query.into(),
            expected: expected.into(),
            cosine_threshold,
        }
    }

    /// Returns the natural language query or prompt.
    pub fn query(&self) -> &str {
        &self.query
    }

    /// Returns the expected textual answer for the case.
    pub fn expected(&self) -> &str {
        &self.expected
    }

    /// Returns the minimum cosine similarity required for success.
    pub fn cosine_threshold(&self) -> f32 {
        self.cosine_threshold
    }
}

/// Collection of related synthetic cases forming a task family.
#[derive(Clone, Debug)]
pub struct SyntheticTask {
    name: String,
    cases: Vec<SyntheticCase>,
}

impl SyntheticTask {
    /// Creates a new synthetic task.
    pub fn new(name: impl Into<String>, cases: Vec<SyntheticCase>) -> Self {
        Self {
            name: name.into(),
            cases,
        }
    }

    /// Returns the machine readable task name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns all cases associated with the task.
    pub fn cases(&self) -> &[SyntheticCase] {
        &self.cases
    }
}

/// Summary metrics for a single task after running the bench.
#[derive(Clone, Debug)]
pub struct TaskReport {
    name: String,
    total_cases: usize,
    correct: usize,
    accuracy: f32,
    average_cosine: f32,
    cosine_threshold_hits: usize,
    within_threshold_ratio: f32,
}

impl TaskReport {
    /// Returns the task identifier.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the number of evaluated cases.
    pub fn total_cases(&self) -> usize {
        self.total_cases
    }

    /// Returns the number of correctly solved cases.
    pub fn correct(&self) -> usize {
        self.correct
    }

    /// Returns the accuracy ratio in the `[0, 1]` range.
    pub fn accuracy(&self) -> f32 {
        self.accuracy
    }

    /// Returns the average cosine similarity produced by the evaluator.
    pub fn average_cosine(&self) -> f32 {
        self.average_cosine
    }

    /// Returns the count of cases that satisfied their cosine thresholds.
    pub fn cosine_threshold_hits(&self) -> usize {
        self.cosine_threshold_hits
    }

    /// Returns the ratio of cases that satisfied the cosine thresholds.
    pub fn within_threshold_ratio(&self) -> f32 {
        self.within_threshold_ratio
    }
}

/// Aggregated metrics across all synthetic tasks.
#[derive(Clone, Debug)]
pub struct BenchReport {
    task_reports: Vec<TaskReport>,
    overall_accuracy: f32,
    overall_average_cosine: f32,
    overall_within_threshold_ratio: f32,
}

impl BenchReport {
    /// Returns the per-task reports.
    pub fn task_reports(&self) -> &[TaskReport] {
        &self.task_reports
    }

    /// Returns the global accuracy across all tasks.
    pub fn overall_accuracy(&self) -> f32 {
        self.overall_accuracy
    }

    /// Returns the average cosine similarity across all evaluated cases.
    pub fn overall_average_cosine(&self) -> f32 {
        self.overall_average_cosine
    }

    /// Returns the ratio of cases meeting their cosine thresholds across the bench.
    pub fn overall_within_threshold_ratio(&self) -> f32 {
        self.overall_within_threshold_ratio
    }
}

/// Synthetic evaluation bench that wires together canonical reasoning challenges.
#[derive(Clone, Debug)]
pub struct SyntheticBench {
    tasks: Vec<SyntheticTask>,
}

impl SyntheticBench {
    /// Builds the default synthetic bench containing XOR, analogies, stack, and retrieval tasks.
    pub fn new() -> Self {
        let tasks = vec![
            SyntheticTask::new(
                "xor",
                vec![
                    SyntheticCase::new("0 0", "0", 0.95),
                    SyntheticCase::new("0 1", "1", 0.95),
                    SyntheticCase::new("1 0", "1", 0.95),
                    SyntheticCase::new("1 1", "0", 0.95),
                ],
            ),
            SyntheticTask::new(
                "analogies",
                vec![
                    SyntheticCase::new("king is to queen as man is to ?", "woman", 0.80),
                    SyntheticCase::new("paris is to france as rome is to ?", "italy", 0.80),
                    SyntheticCase::new("high is to low as hot is to ?", "cold", 0.80),
                ],
            ),
            SyntheticTask::new(
                "stack_parentheses",
                vec![
                    SyntheticCase::new("()[]", "valid", 0.85),
                    SyntheticCase::new("([{}])", "valid", 0.85),
                    SyntheticCase::new("(]", "invalid", 0.85),
                    SyntheticCase::new("((())", "invalid", 0.85),
                ],
            ),
            SyntheticTask::new(
                "long_context_retrieval",
                vec![
                    SyntheticCase::new(
                        "Context: Alpha=red, Beta=green, Gamma=blue. Query: color of Beta?",
                        "green",
                        0.75,
                    ),
                    SyntheticCase::new(
                        "Context: Timeline 1990->Launch, 1995->Orbit, 1999->Return. Query: 1995 event?",
                        "orbit",
                        0.75,
                    ),
                    SyntheticCase::new(
                        "Context: Node42 stores vector embeddings, Node77 stores graphs. Query: Node77 stores?",
                        "graphs",
                        0.75,
                    ),
                ],
            ),
        ];
        Self { tasks }
    }

    /// Returns the tasks contained in the bench.
    pub fn tasks(&self) -> &[SyntheticTask] {
        &self.tasks
    }

    /// Evaluates the bench using the provided evaluator closure.
    pub fn evaluate<F>(&self, mut evaluator: F) -> BenchReport
    where
        F: FnMut(&SyntheticTask, &SyntheticCase) -> (String, f32),
    {
        let mut task_reports = Vec::with_capacity(self.tasks.len());
        let mut total_cases = 0usize;
        let mut total_correct = 0usize;
        let mut total_threshold_hits = 0usize;
        let mut total_cosine = 0.0f32;

        for task in &self.tasks {
            let mut correct = 0usize;
            let mut threshold_hits = 0usize;
            let mut cosine_sum = 0.0f32;

            for case in task.cases() {
                let (prediction, cosine) = evaluator(task, case);
                if prediction.trim() == case.expected() {
                    correct += 1;
                }
                if cosine >= case.cosine_threshold() {
                    threshold_hits += 1;
                }
                cosine_sum += cosine;
            }

            let case_count = task.cases().len();
            let accuracy = if case_count == 0 {
                0.0
            } else {
                correct as f32 / case_count as f32
            };
            let average_cosine = if case_count == 0 {
                0.0
            } else {
                cosine_sum / case_count as f32
            };
            let threshold_ratio = if case_count == 0 {
                0.0
            } else {
                threshold_hits as f32 / case_count as f32
            };

            task_reports.push(TaskReport {
                name: task.name().to_string(),
                total_cases: case_count,
                correct,
                accuracy,
                average_cosine,
                cosine_threshold_hits: threshold_hits,
                within_threshold_ratio: threshold_ratio,
            });

            total_cases += case_count;
            total_correct += correct;
            total_threshold_hits += threshold_hits;
            total_cosine += cosine_sum;
        }

        let overall_accuracy = if total_cases == 0 {
            0.0
        } else {
            total_correct as f32 / total_cases as f32
        };
        let overall_average_cosine = if total_cases == 0 {
            0.0
        } else {
            total_cosine / total_cases as f32
        };
        let overall_within_threshold_ratio = if total_cases == 0 {
            0.0
        } else {
            total_threshold_hits as f32 / total_cases as f32
        };

        BenchReport {
            task_reports,
            overall_accuracy,
            overall_average_cosine,
            overall_within_threshold_ratio,
        }
    }
}
