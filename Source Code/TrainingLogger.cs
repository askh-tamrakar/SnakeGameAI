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
        private static Population population => Program.population;

        private static string logDirectory = "Logs";
        private static string currentSessionLog = "";
        private static string metricsLog = "";
        private static string errorLog = "";
        private static string genomeLog = "";
        private static bool isInitialized = false;

        // Session tracking
        private static DateTime sessionStartTime;
        private static string sessionId = "";

        // Performance metrics
        private static List<GenerationMetrics> generationHistory = new();
        private static List<GenomeLifecycle> genomeLifecycles = new();

        /// <summary>
        /// Metrics tracked per generation
        /// </summary>
        public class GenerationMetrics {
            public int Generation { get; set; }
            public DateTime Timestamp { get; set; }
            public double BestFitness { get; set; }
            public double AverageFitness { get; set; }
            public double SmoothFitness { get; set; }
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
        /// Track individual genome lifecycle
        /// </summary>
        public class GenomeLifecycle {
            public int Generation { get; set; }
            public string GenomeID { get; set; } = "";
            public int StepsSurvived { get; set; }
            public int Score { get; set; }
            public double Fitness { get; set; }
            public string CauseOfDeath { get; set; } = "";
            public int FoodEaten { get; set; }
            public double AvgStepsPerFood { get; set; }
            public int MaxSnakeLength { get; set; }
            public bool AddedToBestList { get; set; }
        }

        /// <summary>
        /// Initialize the logging system
        /// </summary>
        public static void Initialize() {
            if(isInitialized)
                return;

            sessionStartTime = DateTime.Now;
            sessionId = sessionStartTime.ToString("yyyyMMdd_HHmmss");

            logDirectory = Path.Combine(logDirectory, $"session_{sessionId}");

            // Create logs directory if it doesn't exist
            if(!Directory.Exists(logDirectory)) {
                Directory.CreateDirectory(logDirectory);
            }

            // Create log file paths
            currentSessionLog = Path.Combine(logDirectory, $"session_{sessionId}.log");
            metricsLog = Path.Combine(logDirectory, $"metrics_{sessionId}.csv");
            genomeLog = Path.Combine(logDirectory, $"genomes_{sessionId}.csv");
            errorLog = Path.Combine(logDirectory, "errors.log");

            // Initialize session log
            WriteSessionHeader();

            // Initialize CSV Metrics log
            InitializeMetricsCSV();

            // Initialize CSV Genome log
            InitializeGenomeCSV();

            isInitialized = true;

            Log("=== TRAINING LOGGER INITIALIZED ===");
            Log($"Session ID: {sessionId}");
            Log($"Log Directory: {Path.GetFullPath(logDirectory)}");
            Log($"Session Log: {currentSessionLog}");
            Log($"Metrics Log: {metricsLog}");
            Log($"Genomes CSV: {genomeLog}");
            Log("=====================================\n");
        }

        /// <summary>
        /// Write session header with system info
        /// </summary>
        private static void WriteSessionHeader() {
            var sb = new StringBuilder();
            sb.AppendLine("╔════════════════════════════════════════════════════════════════╗");
            sb.AppendLine("║          SNAKE AI - NEUROEVOLUTION TRAINING SESSION            ║");
            sb.AppendLine("╚════════════════════════════════════════════════════════════════╝");
            sb.AppendLine();
            sb.AppendLine($"Session Started: {sessionStartTime:yyyy-MM-dd HH:mm:ss}");
            sb.AppendLine($"Session ID: {sessionId}");
            sb.AppendLine();
            sb.AppendLine("═══ SYSTEM CONFIGURATION ═══");
            sb.AppendLine($"OS: {Environment.OSVersion}");
            sb.AppendLine($".NET Version: {Environment.Version}");
            sb.AppendLine($"Processor Count: {Environment.ProcessorCount}");
            sb.AppendLine($"Working Directory: {Environment.CurrentDirectory}");
            sb.AppendLine();
            sb.AppendLine("═══ HYPERPARAMETERS ═══");
            sb.AppendLine($"Population Size: {Program.population?.Genomes.Count ?? 0}");
            sb.AppendLine($"Network Architecture: {string.Join("-", Program.layerSizes)}");
            sb.AppendLine($"Mutation Rate: {Program.population?.GetMutationRate() ?? 0:F4}");
            sb.AppendLine($"Ghost Mode: {Program.isGhostMode}");
            sb.AppendLine();
            sb.AppendLine("╚═══════════════════════════════════════════════════════════════╝\n");

            File.WriteAllText(currentSessionLog, sb.ToString());
        }

        /// <summary>
        /// Initialize CSV file for metrics tracking - FIXED ALIGNMENT
        /// </summary>
        private static void InitializeMetricsCSV() {
            var headers = new[] {
                "Generation",
                "Timestamp",
                "BestFitness",
                "AvgFitness",
                "SmoothFitness",
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
        /// Initialize per-genome CSV log - FIXED: Removed LifespanSec
        /// </summary>
        private static void InitializeGenomeCSV() {
            var headers = new[] {
                "Generation",
                "GenomeID",
                "StepsSurvived",
                "Score",
                "FoodEaten",
                "Fitness",
                "CauseOfDeath",
                "AvgStepsPerFood",
                "MaxSnakeLength",
                "AddedToBestList"
            };
            File.WriteAllText(genomeLog, string.Join(",", headers) + "\n", Encoding.UTF8);
        }

        /// <summary>
        /// Log general message to session log
        /// </summary>
        public static void Log(string message, LogLevel level = LogLevel.Info) {
            if(!isInitialized)
                Initialize();

            string timestamp = DateTime.Now.ToString("HH:mm:ss.fff");
            string levelStr = level.ToString().ToUpper().PadRight(7);
            string logEntry = $"[{timestamp}] [{levelStr}] {message}";

            // Write to console
            Console.WriteLine(logEntry);

            // Write to file
            try {
                File.AppendAllText(currentSessionLog, logEntry + "\n", Encoding.UTF8);
            } catch(Exception ex) {
                Console.WriteLine($"ERROR: Failed to write to log file: {ex.Message}");
            }
        }

        /// <summary>
        /// Log error with stack trace
        /// </summary>
        public static void LogError(string message, Exception ex = null!) {
            if(!isInitialized)
                Initialize();

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
                File.AppendAllText(errorLog, sb.ToString(), Encoding.UTF8);
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
        /// Log individual genome death/lifecycle
        /// </summary>
        public static void LogGenomeDeath(Genome genome, int generation, bool addedToBest = false) {
            if(!isInitialized)
                Initialize();

            if(genome.GenomeID == population.SmartBestGenome.GenomeID)
                addedToBest = true;

            var lifecycle = new GenomeLifecycle {
                Generation = generation,
                GenomeID = genome.GenomeID,
                StepsSurvived = genome.StepsSnapshot,
                Score = genome.Game.cachedScore,
                FoodEaten = genome.Game.cachedScore,
                Fitness = genome.Fitness,
                CauseOfDeath = genome.Game.causeOfDeath.ToString(),
                MaxSnakeLength = genome.SnakeLength,
                AddedToBestList = addedToBest,
                AvgStepsPerFood = genome.Score > 0 ? (double)genome.StepsSnapshot / genome.Score : 0
            };

            genomeLifecycles.Add(lifecycle);
            WriteGenomeToCSV(lifecycle);
        }

        /// <summary>
        /// Write genome data to CSV - FIXED: Matches header exactly
        /// </summary>
        private static void WriteGenomeToCSV(GenomeLifecycle lifecycle) {
            try {
                var values = new[] {
                    lifecycle.Generation.ToString(),
                    lifecycle.GenomeID,
                    lifecycle.StepsSurvived.ToString(),
                    lifecycle.Score.ToString(),
                    lifecycle.FoodEaten.ToString(),
                    lifecycle.Fitness.ToString("F4"),
                    lifecycle.CauseOfDeath,
                    lifecycle.AvgStepsPerFood.ToString("F3"),
                    lifecycle.MaxSnakeLength.ToString(),
                    lifecycle.AddedToBestList.ToString()
                };

                File.AppendAllText(genomeLog, string.Join(",", values) + "\n", Encoding.UTF8);
            } catch(Exception ex) {
                LogError("Failed to write genome to CSV", ex);
            }
        }

        /// <summary>
        /// Log generation metrics
        /// </summary>
        public static void LogGeneration(Population population) {
            if(!isInitialized)
                Initialize();

            var genStart = DateTime.Now;
            var BestGenomeOfCurrentGeneration = population.BestEverGenomeList.Last() ?? population.SmartBestGenome;

            // Calculate metrics
            var metrics = new GenerationMetrics {
                Generation = population.Generation,
                Timestamp = DateTime.Now,
                BestFitness = BestGenomeOfCurrentGeneration.Fitness,
                AverageFitness = population.AverageFitness,
                SmoothFitness = population.SmoothedFitnessHistory.Last(),
                MedianFitness = CalculateMedian(population.FitnessHistory),
                BestScore = BestGenomeOfCurrentGeneration.CachedScore,
                AverageScore = (int)population.Genomes.Average(g => g.CachedScore),
                AliveCount = population.AliveCount,
                MutationRate = population.GetMutationRate(),
                EliteCount = population.EliteCount,
                BestGenomeID = BestGenomeOfCurrentGeneration.GenomeID ?? "N/A",
                DiversityScore = CalculateDiversity(population),
                MaxStepsSurvived = BestGenomeOfCurrentGeneration.StepsSnapshot,
                AvgStepsSurvived = (int)population.Genomes.Average(g => g.StepsSnapshot),
                GensSinceImprovement = population.GenerationSiceImprovement
            };

            // Calculate improvement
            if(generationHistory.Count > 0) {
                var prevBest = generationHistory.Last().BestFitness;
                metrics.FitnessImprovement = metrics.BestFitness - prevBest;
            }

            // Store in history
            generationHistory.Add(metrics);

            // Log to console/file
            LogGenerationSummary(metrics, population);

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
        private static void LogGenerationSummary(GenerationMetrics metrics, Population population) {
            // Gather all lines (excluding section headings)
            var lines = new List<string> {
                $"║ Time: {metrics.Timestamp:HH:mm:ss}",
                $"║ Best Genome ID: {metrics.BestGenomeID}",
                $"║ Fitness: {metrics.BestFitness:F3} ║ Avg: {metrics.AverageFitness:F2} ║ Median: {metrics.MedianFitness:F2}",
                $"║ Score: {metrics.BestScore} ║ Avg Score: {metrics.AverageScore}",
                $"║ Steps: {metrics.MaxStepsSurvived} ║ Mutation Rate: {metrics.MutationRate:F4}",
                $"║ Diversity: {metrics.DiversityScore:F2}%"
            };

            if(metrics.FitnessImprovement > 0)
                lines.Add($"║ Fitness Improvement: +{metrics.FitnessImprovement:F2}");
            else if(metrics.FitnessImprovement < 0)
                lines.Add($"║ Fitness Decline: {metrics.FitnessImprovement:F2}");

            int eliteCount = population.Genomes.Count / 10;
            int mutatedChildren = population.Genomes.Count - eliteCount;
            var bestGenome = population.BestEverGenome;

            var nextGenLines = new List<string> {
                $"║ Elites Preserved: {eliteCount}",
                $"║ Mutated Children: {mutatedChildren}"
            };

            var bestGenomeLines = new List<string> {
                $"║ Genome ID: {bestGenome.GenomeID}",
                $"║ Score: {bestGenome.Game.cachedScore}",
                $"║ Fitness: {bestGenome.Fitness:F3}",
                $"║ Steps: {bestGenome.StepsSnapshot}"
            };

            // Section headings as you want them
            var headerBar = $"\n╔════════════════════ GENERATION {metrics.Generation} ═════════════════════════╗";
            var nextGenBar = $"╠════════════════════ NEXT GENERATION {metrics.Generation + 1} ════════════════════╣";
            var bestBar = $"╠════════════════════ BEST EVER GENOME ═════════════════════╣";
            var endBar = $"╚{new string('═', Math.Max(headerBar.Length - 2, 40))}╝";

            // Find the longest content or heading line for padding
            int innerWidth = Math.Max(
                lines.Concat(nextGenLines).Concat(bestGenomeLines).Select(l => l.Length).Max(),
                Math.Max(headerBar.Length, Math.Max(nextGenBar.Length, bestBar.Length))
            );

            // Pad all content lines to match the longest length (no right border)
            string Pad(string l) {
                return l.PadRight(innerWidth);
            }

            var sb = new StringBuilder();

            sb.AppendLine(headerBar.PadRight(innerWidth));
            foreach(var l in lines)
                sb.AppendLine(Pad(l));
            sb.AppendLine(nextGenBar.PadRight(innerWidth));
            foreach(var l in nextGenLines)
                sb.AppendLine(Pad(l));
            sb.AppendLine(bestBar.PadRight(innerWidth));
            foreach(var l in bestGenomeLines)
                sb.AppendLine(Pad(l));
            sb.AppendLine(endBar.PadRight(innerWidth));

            Log(sb.ToString(), LogLevel.Info);
        }

        /// <summary>
        /// Write metrics to CSV file - FIXED: Added SmoothFitness
        /// </summary>
        private static void WriteMetricsToCSV(GenerationMetrics metrics) {
            try {
                var values = new[] {
                    metrics.Generation.ToString(),
                    metrics.Timestamp.ToString("yyyy-MM-dd HH:mm:ss"),
                    metrics.BestFitness.ToString("F4"),
                    metrics.AverageFitness.ToString("F4"),
                    metrics.SmoothFitness.ToString("F4"),  // FIXED: Was missing
                    metrics.MedianFitness.ToString("F4"),
                    metrics.BestScore.ToString(),
                    metrics.AverageScore.ToString(),
                    metrics.AliveCount.ToString(),
                    metrics.MutationRate.ToString("F6"),
                    metrics.EliteCount.ToString(),
                    metrics.BestGenomeID,
                    metrics.DiversityScore.ToString("F2"),
                    metrics.GenerationDuration.TotalMilliseconds.ToString("F3"),
                    metrics.MaxStepsSurvived.ToString(),
                    metrics.AvgStepsSurvived.ToString(),
                    metrics.FitnessImprovement.ToString("F4"),
                    metrics.GensSinceImprovement.ToString()
                };

                File.AppendAllText(metricsLog, string.Join(",", values) + "\n", Encoding.UTF8);
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
            sb.AppendLine("╔═══════════════════════════════════════════════════════════════╗");
            sb.AppendLine($"║           DETAILED REPORT - GENERATION {metrics.Generation}                     ║");
            sb.AppendLine("╚═══════════════════════════════════════════════════════════════╝\n");

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

            sb.AppendLine("╚═══════════════════════════════════════════════════════════════╝\n");

            File.WriteAllText(reportPath, sb.ToString(), Encoding.UTF8);
            Log($"Detailed report saved: {reportPath}", LogLevel.Info);
        }

        /// <summary>
        /// Log training summary at the end
        /// </summary>
        public static void LogTrainingSummary() {
            if(!isInitialized || generationHistory.Count == 0)
                return;

            var totalDuration = DateTime.Now - sessionStartTime;
            var bestGen = generationHistory.OrderByDescending(m => m.BestFitness).First();

            var sb = new StringBuilder();
            sb.AppendLine("\n╔════════════════════════════════════════════════════════════════╗");
            sb.AppendLine("║                    TRAINING SESSION SUMMARY                    ║");
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

            sb.AppendLine("╚═══════════════════════════════════════════════════════════════╝\n");
            sb.AppendLine($"Session ended: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            sb.AppendLine($"Logs saved in: {Path.GetFullPath(logDirectory)}");

            string summary = sb.ToString();
            Log(summary, LogLevel.Info);

            // Save summary to separate file
            var summaryPath = Path.Combine(logDirectory, $"summary_{sessionId}.txt");
            File.WriteAllText(summaryPath, summary, Encoding.UTF8);
        }

        /// <summary>
        /// Calculate median from list of values
        /// </summary>
        private static double CalculateMedian(List<double> values) {
            if(values.Count == 0)
                return 0;

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
            var uniqueIDs = population.Genomes.Select(g => g.GenomeID).Distinct().Count();
            return (double)uniqueIDs / population.Genomes.Count * 100.0;
        }

        /// <summary>
        /// Export metrics to CSV for external analysis
        /// </summary>
        public static void ExportMetrics(string filename = null!) {
            if(generationHistory.Count == 0) {
                Log("No metrics to export", LogLevel.Warning);
                return;
            }

            string exportPath = filename ?? Path.Combine(logDirectory, $"export_{sessionId}.csv");
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
