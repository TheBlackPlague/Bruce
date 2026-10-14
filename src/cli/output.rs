use indicatif::{HumanBytes, HumanCount};

use crate::{analyze, convert};

pub struct CommandOutput
{
    pub plain: String,
    pub sections: Vec<OutputSection>,
}

pub struct OutputSection
{
    pub title: &'static str,
    pub fields: Vec<(&'static str, String)>,
}

impl CommandOutput
{
    pub fn conversion(summary: &convert::Summary) -> Self
    {
        let mut fields = vec![("Positions", HumanCount(summary.positions).to_string())];

        if summary.skipped != 0 {
            fields.push(("Skipped", HumanCount(summary.skipped).to_string()));
        }

        fields.push(("Input size", HumanBytes(summary.input_bytes).to_string()));
        fields.push(("Output size", HumanBytes(summary.output_bytes).to_string()));

        Self { plain: summary.to_string(), sections: vec![OutputSection { title: "", fields }] }
    }

    pub fn analysis(report: &analyze::Report) -> Self
    {
        let positions = if report.limit_reached {
            format!("{} (limit reached)", HumanCount(report.positions))
        } else {
            HumanCount(report.positions).to_string()
        };

        let mut scores = vec![("Mean", format!("{:.1} cp", report.scores.mean_cp))];

        if let Some(median) = report.scores.percentiles_cp.get(&50) {
            scores.push(("Median", format!("{median} cp")));
        }

        scores.push(("Range", format!("{} to {} cp", report.scores.minimum_cp, report.scores.maximum_cp)));

        let mut sections = vec![
            OutputSection { title: "", fields: vec![("Positions", positions)] },
            OutputSection { title: "Scores (side to move)", fields: scores },
            OutputSection {
                title: "Outcomes",
                fields: vec![
                    ("Wins", percentage(report.results.wins, report.positions)),
                    ("Draws", percentage(report.results.draws, report.positions)),
                    ("Losses", percentage(report.results.losses, report.positions)),
                ],
            },
        ];

        if let Some(network) = &report.network {
            let errors = &network.overall;

            sections.push(OutputSection {
                title: "Network error",
                fields: vec![
                    ("MAE", format!("{:.1} cp", errors.mean_absolute_error_cp)),
                    ("RMSE", format!("{:.1} cp", errors.root_mean_squared_error_cp)),
                    ("Bias", format!("{:+.1} cp", errors.bias_cp)),
                ],
            });
        }

        Self { plain: report.summary(), sections }
    }

    pub fn check() -> Self { Self { plain: "Configuration and datasets checked".to_owned(), sections: Vec::new() } }
}

fn percentage(count: u64, total: u64) -> String
{
    if total == 0 {
        return "—".to_owned();
    }

    format!("{:.1}%", count as f64 / total as f64 * 100.0)
}

#[cfg(test)]
mod tests
{
    use super::*;
    use crate::architecture::Architecture;

    #[test]
    fn conversion_only_mentions_skipped_positions_when_present()
    {
        let mut summary = convert::Summary { positions: 2048, skipped: 0, input_bytes: 32768, output_bytes: 65536 };
        let output = CommandOutput::conversion(&summary);

        assert_eq!(output.plain, summary.to_string());
        assert!(!output.sections[0].fields.iter().any(|(label, _)| *label == "Skipped"));

        summary.skipped = 12;
        let output = CommandOutput::conversion(&summary);

        assert!(output.sections[0].fields.contains(&("Skipped", "12".to_owned())));
    }

    #[test]
    fn analysis_displays_reported_statistics_and_optional_network_errors()
    {
        let mut report = analyze::Report {
            perspective: "side_to_move",
            positions: 4,
            limit_reached: true,
            scores: analyze::ScoreSummary {
                mean_cp: 25.0,
                minimum_cp: -100,
                maximum_cp: 200,
                percentiles_cp: [(50, 0)].into(),
                ..Default::default()
            },
            score_distribution: Default::default(),
            results: analyze::Results { losses: 1, draws: 1, wins: 2 },
            material: Default::default(),
            network: None,
        };

        let output = CommandOutput::analysis(&report);

        assert_eq!(output.plain, report.summary());
        assert_eq!(output.sections.len(), 3);
        assert!(output.sections[0].fields.contains(&("Positions", "4 (limit reached)".to_owned())));
        assert!(output.sections[1].title.contains("side to move"));
        assert!(output.sections[1].fields.contains(&("Median", "0 cp".to_owned())));
        assert!(output.sections[2].fields.contains(&("Wins", "50.0%".to_owned())));

        let mut errors = analyze::ErrorSummary::default();
        errors.mean_absolute_error_cp = 18.2;
        errors.root_mean_squared_error_cp = 24.7;
        errors.bias_cp = -2.5;

        report.network = Some(analyze::NetworkReport {
            architecture: Architecture::Aurora,
            weights: "native_float_checkpoint",
            score_bucket_cp: 100,
            overall: errors,
            by_teacher_score_cp: Default::default(),
            by_piece_count: Default::default(),
            calibration: Vec::new(),
            extreme_predictions: 0,
            extreme_probability_threshold: 0.01,
        });

        let output = CommandOutput::analysis(&report);
        let network = output.sections.last().unwrap();

        assert_eq!(output.plain, report.summary());
        assert_eq!(network.title, "Network error");
        assert!(network.fields.contains(&("MAE", "18.2 cp".to_owned())));
        assert!(network.fields.contains(&("RMSE", "24.7 cp".to_owned())));
        assert!(network.fields.contains(&("Bias", "-2.5 cp".to_owned())));
    }
}
