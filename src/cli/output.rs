use indicatif::{HumanBytes, HumanCount};

use crate::{analyze, convert};

#[derive(Clone)]
pub struct CommandOutput
{
    pub plain: String,
    pub sections: Vec<OutputSection>,
}

#[derive(Clone)]
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

        if summary.skipped > 0 {
            fields.push(("Skipped", HumanCount(summary.skipped).to_string()));
        }

        fields.extend([
            ("Input size", HumanBytes(summary.input_bytes).to_string()),
            ("Output size", HumanBytes(summary.output_bytes).to_string()),
        ]);

        Self { plain: summary.to_string(), sections: vec![OutputSection { title: "", fields }] }
    }

    pub fn analysis(report: &analyze::Report) -> Self
    {
        let mut positions = HumanCount(report.positions).to_string();
        if report.limit_reached {
            positions.push_str(" (limit reached)");
        }

        let mut scores = vec![("Mean", format!("{:.1} cp", report.scores.mean_cp))];
        if let Some(median) = report.scores.percentiles_cp.get(&50) {
            scores.push(("Median", format!("{median} cp")));
        }
        scores.push(("Range", format!("{} to {} cp", report.scores.minimum_cp, report.scores.maximum_cp)));

        let percentage = |count: u64| {
            if report.positions == 0 {
                "—".into()
            } else {
                format!("{:.1}%", count as f64 * 100.0 / report.positions as f64)
            }
        };

        let mut sections = vec![
            OutputSection { title: "", fields: vec![("Positions", positions)] },
            OutputSection { title: "Scores (side to move)", fields: scores },
            OutputSection {
                title: "Outcomes",
                fields: vec![
                    ("Wins", percentage(report.results.wins)),
                    ("Draws", percentage(report.results.draws)),
                    ("Losses", percentage(report.results.losses)),
                ],
            },
        ];
        if let Some(network) = &report.network {
            let error = &network.overall;
            sections.push(OutputSection {
                title: "Network error",
                fields: vec![
                    ("MAE", format!("{:.1} cp", error.mean_absolute_error_cp)),
                    ("RMSE", format!("{:.1} cp", error.root_mean_squared_error_cp)),
                    ("Bias", format!("{:+.1} cp", error.bias_cp)),
                ],
            });
        }

        Self { plain: report.summary(), sections }
    }

    pub fn check() -> Self { Self { plain: "Configuration and datasets checked".into(), sections: Vec::new() } }
}
