using System.Text.Json;
using RL = Raylib_cs.Raylib;
using MathNet.Numerics.LinearAlgebra;
using static SnakeGameAI.Snake_Game;
using Raylib_cs;

namespace SnakeGameAI {
    public class GenomeDTO {
        public string GenomeID { get; set; } = "";
        public double Fitness { get; set; }
        public int Score { get; set; }
        public int UpdateScore { get; set; }
        public int StepsSnapshot { get; set; }
        public int[] LayerSizes { get; set; } = [];
        public List<MatrixDTO> Weights { get; set; } = [];
        public List<MatrixDTO> Biases { get; set; } = [];
    }

    public class MatrixDTO {
        public int Rows { get; set; }
        public int Columns { get; set; }
        public List<double> Data { get; set; } = [];
    }

    public class SaveWrapper {
        public int GenerationNumber { get; set; }
        public List<double> FitnessHistory { get; set; } = [];
        public List<GenomeDTO> Generation { get; set; } = [];
        public List<GenomeDTO> BestGenomeList { get; set; } = [];
    }

    public static class Genome_Persistence {
        // ⚡ OPTIMIZATION: Static readonly JsonSerializerOptions for reuse
        private static readonly JsonSerializerOptions JsonOptions = new JsonSerializerOptions {
            WriteIndented = true
        };

        // Saves BestEverGenomeList
        public static void SaveGenerationOrBestEverList(Population population, string path) {
            try {
                var dtoBestGenomeList = population.BestEverGenomeList.Select(g => g.ToDTO()).ToList();

                if(RL.IsKeyDown(KeyboardKey.LeftControl)) {
                    var dtoGenerationList = population.Genomes.Select(g => g.ToDTO()).ToList();
                    var saveWrapper = new SaveWrapper {
                        GenerationNumber = population.Generation,
                        FitnessHistory = population.FitnessHistory.ToList(),
                        Generation = dtoGenerationList,
                        BestGenomeList = dtoBestGenomeList
                    };

                    // ⚡ OPTIMIZATION: Reuse JsonOptions
                    var json = JsonSerializer.Serialize(saveWrapper, JsonOptions);
                    File.WriteAllText("Save States/Generation.json", json);
                    Console.WriteLine("✓ Saved Generation + Best Genomes");
                } else {
                    var json = JsonSerializer.Serialize(dtoBestGenomeList, JsonOptions);
                    File.WriteAllText(path, json);
                    Console.WriteLine($"✓ Saved Best Genomes to {path}");
                }
            } catch(Exception ex) {
                Console.WriteLine("❌ Save failed: " + ex.Message);
            }
        }

        public static (List<Genome> Generation, List<Genome> BestList)
        LoadGenerationOrBestEverList(Population population, string path) {
            if(!File.Exists(path)) {
                Console.WriteLine("❌ Load failed: File does not exist.");
                return (new List<Genome>(), new List<Genome>());
            }

            try {
                var json = File.ReadAllText("Save States/Generation.json");

                if(json.TrimStart().StartsWith("{")) {
                    var wrapper = JsonSerializer.Deserialize<SaveWrapper>(json);
                    var generation = wrapper?.Generation?.Select(dto => dto.ToGenome()).ToList() ?? [];
                    var bestList = wrapper?.BestGenomeList?.Select(dto => dto.ToGenome()).ToList() ?? [];
                    population.Generation = wrapper!.GenerationNumber;
                    population.FitnessHistory = wrapper.FitnessHistory;
                    Console.WriteLine("✓ Loaded Generation + BestList.");
                    return (generation, bestList);
                } else {
                    json = File.ReadAllText(path);
                    var dtoList = JsonSerializer.Deserialize<List<GenomeDTO>>(json);
                    var bestList = dtoList?.Select(dto => dto.ToGenome()).ToList() ?? [];
                    Console.WriteLine("✓ Loaded BestEver List.");
                    return (new List<Genome>(), bestList);
                }
            } catch(Exception ex) {
                Console.WriteLine($"❌ Load error: {ex.Message}");
                return (new List<Genome>(), new List<Genome>());
            }
        }

        // ⚡ OPTIMIZATION: Streamlined DTO conversion
        public static GenomeDTO ToDTO(this Genome genome) {
            var neuralNetwork = genome.NeuralNetwork;
            int[] layerSizes = Program.layerSizes;

            return new GenomeDTO {
                GenomeID = genome.GenomeID,
                Fitness = genome.Fitness,
                Score = genome.Score,
                UpdateScore = genome.Game.cachedScore,
                StepsSnapshot = genome.StepsSnapshot,
                LayerSizes = layerSizes,
                Weights = neuralNetwork.GetAllWeights().Select(m => new MatrixDTO {
                    Rows = m.RowCount,
                    Columns = m.ColumnCount,
                    Data = m.ToColumnMajorArray().ToList()
                }).ToList(),
                Biases = neuralNetwork.GetAllBiases().Select(m => new MatrixDTO {
                    Rows = m.RowCount,
                    Columns = m.ColumnCount,
                    Data = m.ToColumnMajorArray().ToList()
                }).ToList()
            };
        }

        public static Genome ToGenome(this GenomeDTO dto) {
            var genome = new Genome(dto.LayerSizes);
            genome.GenomeID = dto.GenomeID;
            genome.Fitness = dto.Fitness;
            genome.Game.score = dto.UpdateScore;
            genome.StepsSnapshot = dto.StepsSnapshot;

            var weights = dto.Weights.Select(m =>
                Matrix<double>.Build.Dense(m.Rows, m.Columns, m.Data.ToArray())).ToList();
            var biases = dto.Biases.Select(m =>
                Matrix<double>.Build.Dense(m.Rows, m.Columns, m.Data.ToArray())).ToList();

            var neuralNetwork = genome.NeuralNetwork;
            var weightList = neuralNetwork.GetAllWeights();
            var biasesList = neuralNetwork.GetAllBiases();

            for(int i = 0; i < weightList.Count; i++)
                weightList[i].SetSubMatrix(0, 0, weights[i]);
            for(int i = 0; i < biasesList.Count; i++)
                biasesList[i].SetSubMatrix(0, 0, biases[i]);

            return genome;
        }
    }
}
