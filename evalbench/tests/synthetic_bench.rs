use evalbench::synthetic::{SyntheticBench, SyntheticTask, TaskReport};

fn bench_task_names(bench: &SyntheticBench) -> Vec<&str> {
    bench.tasks().iter().map(SyntheticTask::name).collect()
}

fn approx_eq(a: f32, b: f32) {
    assert!((a - b).abs() < 1e-5, "{a} != {b}");
}

#[test]
fn synthetic_bench_provides_expected_task_suite() {
    let bench = SyntheticBench::new();
    let mut names = bench_task_names(&bench);
    names.sort();

    assert_eq!(
        names,
        vec![
            "analogies",
            "long_context_retrieval",
            "stack_parentheses",
            "xor"
        ],
    );

    let xor = bench
        .tasks()
        .iter()
        .find(|task| task.name() == "xor")
        .expect("xor task present");
    assert_eq!(xor.cases().len(), 4);

    let analogies = bench
        .tasks()
        .iter()
        .find(|task| task.name() == "analogies")
        .expect("analogies task present");
    assert_eq!(analogies.cases().len(), 3);

    let stack = bench
        .tasks()
        .iter()
        .find(|task| task.name() == "stack_parentheses")
        .expect("stack task present");
    assert_eq!(stack.cases().len(), 4);

    let retrieval = bench
        .tasks()
        .iter()
        .find(|task| task.name() == "long_context_retrieval")
        .expect("retrieval task present");
    assert_eq!(retrieval.cases().len(), 3);
}

#[test]
fn synthetic_bench_reports_perfect_evaluator_metrics() {
    let bench = SyntheticBench::new();
    let report =
        bench.evaluate(|_, case| (case.expected().to_string(), case.cosine_threshold() + 0.05));

    assert_eq!(report.task_reports().len(), bench.tasks().len());
    for (task, task_report) in bench.tasks().iter().zip(report.task_reports()) {
        let expected_avg = task
            .cases()
            .iter()
            .map(|case| case.cosine_threshold() + 0.05)
            .sum::<f32>()
            / task.cases().len() as f32;
        approx_eq(task_report.accuracy(), 1.0);
        approx_eq(task_report.average_cosine(), expected_avg);
        approx_eq(task_report.within_threshold_ratio(), 1.0);
    }

    approx_eq(report.overall_accuracy(), 1.0);
    approx_eq(report.overall_within_threshold_ratio(), 1.0);
    assert!(report.overall_average_cosine() > 0.0);
}

#[test]
fn synthetic_bench_handles_partial_success_metrics() {
    let bench = SyntheticBench::new();
    let report = bench.evaluate(|task, case| match task.name() {
        "xor" => match case.query() {
            "0 0" | "0 1" | "1 0" => (case.expected().to_string(), 0.96),
            "1 1" => ("1".to_string(), 0.40),
            _ => unreachable!(),
        },
        "analogies" => match case.query() {
            "king is to queen as man is to ?" => ("empress".to_string(), 0.50),
            "paris is to france as rome is to ?" => (
                case.expected().to_string(),
                0.82,
            ),
            "high is to low as hot is to ?" => (case.expected().to_string(), 0.82),
            _ => unreachable!(),
        },
        "stack_parentheses" => match case.query() {
            "()[]" | "([{}])" => (case.expected().to_string(), 0.90),
            "(]" | "((())" => (case.expected().to_string(), 0.83),
            _ => unreachable!(),
        },
        "long_context_retrieval" => match case.query() {
            "Context: Alpha=red, Beta=green, Gamma=blue. Query: color of Beta?" => (
                case.expected().to_string(),
                0.78,
            ),
            "Context: Timeline 1990->Launch, 1995->Orbit, 1999->Return. Query: 1995 event?" => (
                "landing".to_string(),
                0.70,
            ),
            "Context: Node42 stores vector embeddings, Node77 stores graphs. Query: Node77 stores?" => (
                case.expected().to_string(),
                0.80,
            ),
            _ => unreachable!(),
        },
        other => panic!("unexpected task {other}"),
    });

    let task = |name: &str| -> &TaskReport {
        report
            .task_reports()
            .iter()
            .find(|task| task.name() == name)
            .expect("task report present")
    };

    let xor = task("xor");
    approx_eq(xor.accuracy(), 0.75);
    approx_eq(xor.average_cosine(), 0.82);
    approx_eq(xor.within_threshold_ratio(), 0.75);

    let analogies = task("analogies");
    approx_eq(analogies.accuracy(), 2.0 / 3.0);
    approx_eq(analogies.average_cosine(), 2.14 / 3.0);
    approx_eq(analogies.within_threshold_ratio(), 2.0 / 3.0);

    let stack = task("stack_parentheses");
    approx_eq(stack.accuracy(), 1.0);
    approx_eq(stack.average_cosine(), 3.46 / 4.0);
    approx_eq(stack.within_threshold_ratio(), 0.5);

    let retrieval = task("long_context_retrieval");
    approx_eq(retrieval.accuracy(), 2.0 / 3.0);
    approx_eq(retrieval.average_cosine(), 2.28 / 3.0);
    approx_eq(retrieval.within_threshold_ratio(), 2.0 / 3.0);

    approx_eq(report.overall_accuracy(), 11.0 / 14.0);
    approx_eq(report.overall_average_cosine(), 11.16 / 14.0);
    approx_eq(report.overall_within_threshold_ratio(), 9.0 / 14.0);
}
