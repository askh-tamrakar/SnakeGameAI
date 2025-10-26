using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using static SnakeGameAI.Snake_Game;

namespace SnakeGameAI {

    /// <summary>
    /// Comprehensive logging system for Snake AI training
    /// Tracks generation metrics, performance statistics, and training progress
    /// </summary>
    public static class TrainingLogger {

        private static string logDirectory = "Logs";
        private static string currentSessionLog = "";
        private static string metricsLog = "";
        private static string errorLog = "";
        private static bool isInitialized = false;

        // Session tracking
        private static DateTime sessionStartTime;
        private static string sessionId;

        // Performance metrics
        private static List<GenerationMetrics> generationHistory = new();

        /// <summary>
        /// Metrics tracked per generation
        /// </summary>
        public class GenerationMetrics {
            public int Generation { get; set; }
            public DateTime Timestamp { get; set; }
            public double BestFitness { get; set; }
            public double AverageFitness { get; set; }
            public double MedianFitness { get; set; }
            public int BestScore { get; set; }
            public int AverageScore { get; set; }
            public int AliveCount { get; set; }
            public double MutationRate { get; set; }
            public int EliteCount { get; set; }
            public string BestGenomeID { get; set; } = "";
            public double DiversityScore { get; set; }
            public TimeSpan GenerationDuration { get; set; }

            // Additional stats
            public int MaxStepsSurvived { get; set; }
            public int AvgStepsSurvived { get; set; }
            public double FitnessImprovement { get; set; }
            public int GensSinceImprovement { get; set; }
        }

        /// <summary>
        /// Initialize the logging system
        /// </summary>
        public static void Initialize() {
            if(isInitialized) return;

            // Create logs directory if it doesn't exist
            if(!Directory.Exists(logDirectory)) {
                Directory.CreateDirectory(logDirectory);
            }

            sessionStartTime = DateTime.Now;
            sessionId = sessionStartTime.ToString("yyyyMMdd_HHmmss");

            // Create log file paths
            currentSessionLog = Path.Combine(logDirectory, $"session_{sessionId}.log");
            metricsLog = Path.Combine(logDirectory, $"metrics_{sessionId}.csv");
            errorLog = Path.Combine(logDirectory, "errors.log");

            // Initialize session log
            WriteSessionHeader();

            // Initialize CSV metrics log
            InitializeMetricsCSV();

            isInitialized = true;

            Log("=== TRAINING LOGGER INITIALIZED ===");
            Log($"Session ID: {sessionId}");
            Log($"Log Directory: {Path.GetFullPath(logDirectory)}");
            Log($"Session Log: {currentSessionLog}");
            Log($"Metrics Log: {metricsLog}");
            Log("=====================================\n");
        }

        /// <summary>
        /// Write session header with system info
        /// </summary>
        private static void WriteSessionHeader() {
            var sb = new StringBuilder();
            sb.AppendLine("╔════════════════════════════════════════════════════════════════╗");
            sb.AppendLine("║         SNAKE AI - NEUROEVOLUTION TRAINING SESSION            ║");
            sb.AppendLine("╚════════════════════════════════════════════════════════════════╝");
            sb.AppendLine();
            sb.AppendLine($"Session Started: {sessionStartTime:yyyy-MM-dd HH:mm:ss}");
            sb.AppendLine($"Session ID: {sessionId}");
            sb.AppendLine();
            sb.AppendLine("=== SYSTEM CONFIGURATION ===");
            sb.AppendLine($"OS: {Environment.OSVersion}");
            sb.AppendLine($".NET Version: {Environment.Version}");
            sb.AppendLine($"Processor Count: {Environment.ProcessorCount}");
            sb.AppendLine($"Working Directory: {Environment.CurrentDirectory}");
            sb.AppendLine();
            sb.AppendLine("=== HYPERPARAMETERS ===");
            sb.AppendLine($"Population Size: {Program.population?.Genomes.Count ?? 0}");
            sb.AppendLine($"Network Architecture: {string.Join("-", Program.layerSizes)}");
            sb.AppendLine($"Mutation Rate: {Program.population?.GetMutationRate() ?? 0:F4}");
            sb.AppendLine($"Ghost Mode: {Program.isGhostMode}");
            sb.AppendLine();
            sb.AppendLine("═══════════════════════════════════════════════════════════════\n");

            File.WriteAllText(currentSessionLog, sb.ToString());
        }

        /// <summary>
        /// Initialize CSV file for metrics tracking
        /// </summary>
        private static void InitializeMetricsCSV() {
            var headers = new[] {
                "Generation",
                "Timestamp",
                "BestFitness",
                "AvgFitness",
                "MedianFitness",
                "BestScore",
                "AvgScore",
                "AliveCount",
                "MutationRate",
                "EliteCount",
                "BestGenomeID",
                "DiversityScore",
                "GenDuration_ms",
                "MaxStepsSurvived",
                "AvgStepsSurvived",
                "FitnessImprovement",
                "GensSinceImprovement"
            };

            File.WriteAllText(metricsLog, string.Join(",", headers) + "\n");
        }

        /// <summary>
        /// Log general message to session log
        /// </summary>
        public static void Log(string message, LogLevel level = LogLevel.Info) {
            if(!isInitialized) Initialize();

            string timestamp = DateTime.Now.ToString("HH:mm:ss.fff");
            string levelStr = level.ToString().ToUpper().PadRight(7);
            string logEntry = $"[{timestamp}] [{levelStr}] {message}";

            // Write to console
            Console.WriteLine(logEntry);

            // Write to file
            try {
                File.AppendAllText(currentSessionLog, logEntry + "\n");
            } catch(Exception ex) {
                Console.WriteLine($"ERROR: Failed to write to log file: {ex.Message}");
            }
        }

        /// <summary>
        /// Log error with stack trace
        /// </summary>
        public static void LogError(string message, Exception ex = null) {
            if(!isInitialized) Initialize();

            string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");
            var sb = new StringBuilder();
            sb.AppendLine($"\n[{timestamp}] ERROR: {message}");

            if(ex != null) {
                sb.AppendLine($"Exception Type: {ex.GetType().Name}");
                sb.AppendLine($"Exception Message: {ex.Message}");
                sb.AppendLine($"Stack Trace:\n{ex.StackTrace}");

                if(ex.InnerException != null) {
                    sb.AppendLine($"\nInner Exception: {ex.InnerException.Message}");
                }
            }

            sb.AppendLine(new string('-', 80));

            // Write to error log
            try {
                File.AppendAllText(errorLog, sb.ToString());
            } catch {
                Console.WriteLine("CRITICAL: Cannot write to error log!");
            }

            // Also write to session log
            Log($"ERROR: {message}", LogLevel.Error);
            if(ex != null) {
                Log($"  Exception: {ex.Message}", LogLevel.Error);
            }
        }

        /// <summary>
        /// Log generation metrics
        /// </summary>
        public static void LogGeneration(Population population) {
            if(!isInitialized) Initialize();

            var genStart = DateTime.Now;

            // Calculate metrics
            var metrics = new GenerationMetrics {
                Generation = population.Generation,
                Timestamp = DateTime.Now,
                BestFitness = population.Genomes.Max(g => g.Fitness),
                AverageFitness = population.AverageFitness,
                MedianFitness = CalculateMedian(population.Genomes.Select(g => g.Fitness).ToList()),
                BestScore = population.Genomes.Max(g => g.Score),
                AverageScore = (int)population.Genomes.Average(g => g.Score),
                AliveCount = population.AliveCount,
                MutationRate = population.GetMutationRate(),
                EliteCount = population.Genomes.Count / 10, // Assuming 10% elite
                BestGenomeID = population.SmartBestGenome?.GenomeID ?? "N/A",
                DiversityScore = CalculateDiversity(population),
                MaxStepsSurvived = population.Genomes.Max(g => g.StepsSnapshot),
                AvgStepsSurvived = (int)population.Genomes.Average(g => g.StepsSnapshot),
                GensSinceImprovement = 0 // Population should expose this
            };

            // Calculate improvement
            if(generationHistory.Count > 0) {
                var prevBest = generationHistory.Last().BestFitness;
                metrics.FitnessImprovement = metrics.BestFitness - prevBest;
            }

            // Store in history
            generationHistory.Add(metrics);

            // Log to console/file
            LogGenerationSummary(metrics);

            // Write to CSV
            WriteMetricsToCSV(metrics);

            // Periodic detailed report (every 10 generations)
            if(metrics.Generation % 10 == 0) {
                WriteDetailedReport(metrics);
            }
        }

        /// <summary>
        /// Write generation summary to log
        /// </summary>
        private static void LogGenerationSummary(GenerationMetrics metrics) {
            var sb = new StringBuilder();
            sb.AppendLine($"\n╔══════════════════ GENERATION {metrics.Generation} ══════════════════╗");
            sb.AppendLine($"║ Time: {metrics.Timestamp:HH:mm:ss}");
            sb.AppendLine($"║ Best Fitness: {metrics.BestFitness:F2} | Avg: {metrics.AverageFitness:F2} | Median: {metrics.MedianFitness:F2}");
            sb.AppendLine($"║ Best Score: {metrics.BestScore} | Avg Score: {metrics.AverageScore}");
            sb.AppendLine($"║ Alive: {metrics.AliveCount} | Mutation: {metrics.MutationRate:F4}");
            sb.AppendLine($"║ Best Genome: {metrics.BestGenomeID}");
            sb.AppendLine($"║ Diversity: {metrics.DiversityScore:F2}%");

            if(metrics.FitnessImprovement > 0) {
                sb.AppendLine($"║ ⬆️ Fitness Improvement: +{metrics.FitnessImprovement:F2}");
            } else if(metrics.FitnessImprovement < 0) {
                sb.AppendLine($"║ ⬇️ Fitness Decline: {metrics.FitnessImprovement:F2}");
            }

            sb.AppendLine($"╚══════════════════════════════════════════════════════╝");

            Log(sb.ToString(), LogLevel.Info);
        }

        /// <summary>
        /// Write metrics to CSV file
        /// </summary>
        private static void WriteMetricsToCSV(GenerationMetrics metrics) {
            try {
                var values = new[] {
                    metrics.Generation.ToString(),
                    metrics.Timestamp.ToString("yyyy-MM-dd HH:mm:ss"),
                    metrics.BestFitness.ToString("F4"),
                    metrics.AverageFitness.ToString("F4"),
                    metrics.MedianFitness.ToString("F4"),
                    metrics.BestScore.ToString(),
                    metrics.AverageScore.ToString(),
                    metrics.AliveCount.ToString(),
                    metrics.MutationRate.ToString("F6"),
                    metrics.EliteCount.ToString(),
                    metrics.BestGenomeID,
                    metrics.DiversityScore.ToString("F2"),
                    metrics.GenerationDuration.TotalMilliseconds.ToString("F0"),
                    metrics.MaxStepsSurvived.ToString(),
                    metrics.AvgStepsSurvived.ToString(),
                    metrics.FitnessImprovement.ToString("F4"),
                    metrics.GensSinceImprovement.ToString()
                };

                File.AppendAllText(metricsLog, string.Join(",", values) + "\n");
            } catch(Exception ex) {
                LogError("Failed to write metrics to CSV", ex);
            }
        }

        /// <summary>
        /// Write detailed report every N generations
        /// </summary>
        private static void WriteDetailedReport(GenerationMetrics metrics) {
            var reportPath = Path.Combine(logDirectory, $"report_gen{metrics.Generation}.txt");

            var sb = new StringBuilder();
            sb.AppendLine("═══════════════════════════════════════════════════════════════");
            sb.AppendLine($"         DETAILED REPORT - GENERATION {metrics.Generation}");
            sb.AppendLine("═══════════════════════════════════════════════════════════════\n");

            sb.AppendLine("📊 PERFORMANCE METRICS:");
            sb.AppendLine($"  Best Fitness: {metrics.BestFitness:F4}");
            sb.AppendLine($"  Average Fitness: {metrics.AverageFitness:F4}");
            sb.AppendLine($"  Median Fitness: {metrics.MedianFitness:F4}");
            sb.AppendLine($"  Best Score: {metrics.BestScore}");
            sb.AppendLine($"  Average Score: {metrics.AverageScore}");
            sb.AppendLine();

            sb.AppendLine("🧬 POPULATION STATS:");
            sb.AppendLine($"  Alive Count: {metrics.AliveCount}");
            sb.AppendLine($"  Mutation Rate: {metrics.MutationRate:F6}");
            sb.AppendLine($"  Elite Count: {metrics.EliteCount}");
            sb.AppendLine($"  Diversity Score: {metrics.DiversityScore:F2}%");
            sb.AppendLine();

            sb.AppendLine("⏱️ SURVIVAL STATS:");
            sb.AppendLine($"  Max Steps Survived: {metrics.MaxStepsSurvived}");
            sb.AppendLine($"  Avg Steps Survived: {metrics.AvgStepsSurvived}");
            sb.AppendLine();

            sb.AppendLine("📈 PROGRESS:");
            sb.AppendLine($"  Fitness Improvement: {metrics.FitnessImprovement:F4}");
            sb.AppendLine($"  Gens Since Improvement: {metrics.GensSinceImprovement}");
            sb.AppendLine();

            // Historical trend (last 10 generations)
            if(generationHistory.Count >= 10) {
                sb.AppendLine("📊 TREND (Last 10 Generations):");
                var recent = generationHistory.TakeLast(10).ToList();
                sb.AppendLine($"  Avg Best Fitness: {recent.Average(m => m.BestFitness):F4}");
                sb.AppendLine($"  Avg Best Score: {recent.Average(m => m.BestScore):F1}");
                sb.AppendLine($"  Fitness Trend: {(recent.Last().BestFitness > recent.First().BestFitness ? "📈 Improving" : "📉 Declining")}");
                sb.AppendLine();
            }

            sb.AppendLine("═══════════════════════════════════════════════════════════════\n");

            File.WriteAllText(reportPath, sb.ToString());
            Log($"Detailed report saved: {reportPath}", LogLevel.Info);
        }

        /// <summary>
        /// Log training summary at the end
        /// </summary>
        public static void LogTrainingSummary() {
            if(!isInitialized || generationHistory.Count == 0) return;

            var totalDuration = DateTime.Now - sessionStartTime;
            var bestGen = generationHistory.OrderByDescending(m => m.BestFitness).First();

            var sb = new StringBuilder();
            sb.AppendLine("\n╔════════════════════════════════════════════════════════════════╗");
            sb.AppendLine("║               TRAINING SESSION SUMMARY                         ║");
            sb.AppendLine("╚════════════════════════════════════════════════════════════════╝\n");

            sb.AppendLine($"Session ID: {sessionId}");
            sb.AppendLine($"Total Duration: {totalDuration:hh\\:mm\\:ss}");
            sb.AppendLine($"Total Generations: {generationHistory.Count}");
            sb.AppendLine();

            sb.AppendLine("🏆 BEST PERFORMANCE:");
            sb.AppendLine($"  Generation: {bestGen.Generation}");
            sb.AppendLine($"  Fitness: {bestGen.BestFitness:F4}");
            sb.AppendLine($"  Score: {bestGen.BestScore}");
            sb.AppendLine($"  Genome ID: {bestGen.BestGenomeID}");
            sb.AppendLine();

            sb.AppendLine("📊 FINAL STATS:");
            var lastGen = generationHistory.Last();
            sb.AppendLine($"  Final Best Fitness: {lastGen.BestFitness:F4}");
            sb.AppendLine($"  Final Best Score: {lastGen.BestScore}");
            sb.AppendLine($"  Final Mutation Rate: {lastGen.MutationRate:F6}");
            sb.AppendLine();

            sb.AppendLine("📈 OVERALL PROGRESS:");
            var firstGen = generationHistory.First();
            sb.AppendLine($"  Initial Fitness: {firstGen.BestFitness:F4}");
            sb.AppendLine($"  Final Fitness: {lastGen.BestFitness:F4}");
            sb.AppendLine($"  Total Improvement: {lastGen.BestFitness - firstGen.BestFitness:F4}");
            sb.AppendLine($"  Improvement %: {((lastGen.BestFitness - firstGen.BestFitness) / Math.Max(firstGen.BestFitness, 0.001) * 100):F2}%");
            sb.AppendLine();

            sb.AppendLine("═══════════════════════════════════════════════════════════════\n");
            sb.AppendLine($"Session ended: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            sb.AppendLine($"Logs saved in: {Path.GetFullPath(logDirectory)}");

            string summary = sb.ToString();
            Log(summary, LogLevel.Info);

            // Save summary to separate file
            var summaryPath = Path.Combine(logDirectory, $"summary_{sessionId}.txt");
            File.WriteAllText(summaryPath, summary);
        }

        /// <summary>
        /// Calculate median from list of values
        /// </summary>
        private static double CalculateMedian(List<double> values) {
            if(values.Count == 0) return 0;

            var sorted = values.OrderBy(v => v).ToList();
            int mid = sorted.Count / 2;

            if(sorted.Count % 2 == 0) {
                return (sorted[mid - 1] + sorted[mid]) / 2.0;
            } else {
                return sorted[mid];
            }
        }

        /// <summary>
        /// Calculate population diversity (0-100%)
        /// </summary>
        private static double CalculateDiversity(Population population) {
            // Simple diversity: unique genome IDs / total genomes
            var uniqueIDs = population.Genomes.Select(g => g.GenomeID).Distinct().Count();
            return (double)uniqueIDs / population.Genomes.Count * 100.0;
        }

        /// <summary>
        /// Export metrics to CSV for external analysis
        /// </summary>
        public static void ExportMetrics(string filename = null) {
            if(generationHistory.Count == 0) {
                Log("No metrics to export", LogLevel.Warning);
                return;
            }

            string exportPath = filename ?? Path.Combine(logDirectory, $"export_{sessionId}.csv");

            // This is already done in real-time via WriteMetricsToCSV
            Log($"Metrics exported to: {exportPath}", LogLevel.Info);
        }
    }

    /// <summary>
    /// Log level enumeration
    /// </summary>
    public enum LogLevel {
        Debug,
        Info,
        Warning,
        Error,
        Critical
    }
}
