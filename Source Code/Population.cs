using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using Raylib_cs;
using System.Numerics;
using static SnakeGameAI.Program;
using static SnakeGameAI.Snake_Game;
using Color = Raylib_cs.Color;

namespace SnakeGameAI {
    public class Population {
        private const int EliteFraction = 10;

        // ⚡ OPTIMIZATION: Thread-local Random for better performance
        [ThreadStatic]
        private static Random? _threadRandom;
        private static Random GetRandom() => _threadRandom ??= new Random(Guid.NewGuid().GetHashCode());

        public List<double> FitnessHistory { get; set; } = [];
        public List<double> SmoothedFitnessHistory { get; private set; } = [];
        public List<double> AverageFitnessHistory { get; } = [];
        public List<Genome> BestEverGenomeList { get; set; } = [];
        public List<Genome> DeadGenomes { get; set; } = [];
        public List<Genome> Genomes { get; set; } = [];

        private int generation = 0;
        private int eliteCount;

        private double minMutationRate = 0.003d;  
        private double mutationRate = 0.02d;     
        private double maxMutationRate = 0.25d;   
        private double plateauThreshold = 5;     
        private int gensSinceImprovement = 0;
        private double previousBestFitness = 0;

        public int Generation {
            get { return generation; }
            set { generation = value; }
        }

        public int AliveCount => Genomes.Count(g => g.Game.isRunning);
        public double AverageFitness => avgFitness;
        private double avgFitness;
        private Genome bestEverGenome = null!;
        public Genome FinalGenomeFromList { get; set; } = null!;

        public Population(int size, int[] layerSizes) {
            // ⚡ OPTIMIZATION: Pre-allocate capacity
            Genomes = new List<Genome>(size);

            for(int i = 0; i < size; i++) {
                var genome = new Genome(layerSizes);
                genome.AssignID(Generation + 2560, i + 2744);
                Genomes.Add(genome);
            }
        }

        public void UpdateAllGenomes(bool isTraining = true) {
            if(Raylib.IsKeyPressed(KeyboardKey.A)) {
                bool isManualUpdate = true;
                UpdateBestEverGenome(isManualUpdate);
            }

            List<Genome> UpdateGenomes = isTraining ? Genomes : BestEverGenomeList;

            foreach(var genome in UpdateGenomes) {
                if(genome.IsSnakeDead) {
                    genome.Game.ClearFood();
                    continue;
                }

                var input = genome.Game.GetInputs();
                Matrix<double> inputMatrix = Matrix<double>.Build.Dense(1, input.Length, (i, j) => input[j]);
                var (_, move) = genome.NeuralNetwork.Predict(inputMatrix);

                // Apply move
                Vector2 currentDir = genome.Game.snake.direction;
                Vector2 left = new(-currentDir.Y, currentDir.X);
                Vector2 right = new(currentDir.Y, -currentDir.X);

                switch(move) {
                case 0:
                    break;
                case 1:
                    genome.Game.snake.direction = left;
                    break;
                case 2:
                    genome.Game.snake.direction = right;
                    break;
                }

                genome.Game.Update();
                if(!isGraphShowing)
                    genome.Game.Draw();
            }

            string logText = $"Gen: {Generation} | " +
                $"Avg: {AverageFitness:F3} | " +
                $"Alive: {AliveCount} | " +
                $"Best ID: {SmartBestGenome.GenomeID}";
            Raylib.DrawTextEx(font, logText, new Vector2(380 - offset, offset + cellSize * cellCount + 10), 40, 0, Color.Black);
        }

        // ⚡ OPTIMIZATION: Pre-allocated lists, better iteration
        public void Evolve() {
            ApplyFitnessSharing(Genomes);

            Genomes = Genomes.OrderByDescending(g => g.Fitness)
                             .ThenByDescending(g => g.Game.updateScore)
                             .ToList();

            eliteCount = Math.Max(1, Genomes.Count / EliteFraction);

            // ⚡ OPTIMIZATION: Pre-allocate list capacity
            var elites = new List<Genome>(eliteCount);
            for(int i = 0; i < eliteCount; i++) {
                elites.Add(Genomes[i].ShallowClone());
            }

            List<Genome> newGen = elites;

            var viableGenomes = Genomes.Where(g => g.Fitness > 5).ToList();
            if(viableGenomes.Count < eliteCount) {
                viableGenomes = Genomes.Take(eliteCount).ToList();
            }

            for(int i = 0; i < newGen.Count; i++) {
                newGen[i].AssignID(Generation + 2561, i + 2744);
            }

            while(newGen.Count < Genomes.Count) {
                int tournamentSize = 8;
                var parent1 = SelectParent(viableGenomes, tournamentSize);
                var parent2 = SelectParent(viableGenomes, tournamentSize);
                var child = Crossover(parent1, parent2, mutationRate);
                int index = newGen.Count;
                child.AssignID(Generation + 2561, index + 2744);
                newGen.Add(child);
            }

            Genomes = newGen;
        }

        public void EvolveBestEverGenomes() {
            BestEverGenomeList = BestEverGenomeList
                .OrderByDescending(g => g.Fitness)
                .ThenByDescending(g => g.Game.updateScore)
                .ToList();

            for(int i = 0; i < BestEverGenomeList.Count; i++) {
                var parent = BestEverGenomeList[i];
                var child = parent.CloneAndMutate(mutationRate);
                BestEverGenomeList[i] = child;
            }
        }

        public void UpdateBestEverGenome(bool isManualUpdate = false) {
            Console.WriteLine($"Trying to add/update best genome...");

            DeadGenomes = DeadGenomes.OrderByDescending(g => g.Game.updateScore)
                .ThenByDescending(g => g.Fitness)
                .ToList();

            double minFitness = DeadGenomes.Min(g => g.Fitness);
            double shift = (minFitness < 0) ? -minFitness : 0;
            avgFitness = DeadGenomes.Average(g => g.Fitness + shift);
            AverageFitnessHistory.Add(avgFitness);

            Genome bestGenome = DeadGenomes.First();
            DeadGenomes.Clear();

            if(isManualUpdate) {
                bestGenome.Game.updateScore = bestGenome.Score;
                bestGenome.CalculateFitness();
            }

            int existingIndex = BestEverGenomeList.FindIndex(g => g.GenomeID == bestGenome.GenomeID);
            bool exists = existingIndex != -1;

            bool isNewBest = bestEverGenome == null ||
                bestGenome.Score > bestEverGenome.Score ||
                (bestGenome.Score == bestEverGenome.Score && bestGenome.Fitness > bestEverGenome.Fitness);

            if(isNewBest) {
                previousBestFitness = bestEverGenome?.Fitness ?? 0;
                bestEverGenome = bestGenome.DeepClone();
                UpdateMutationRate(bestGenome.Fitness);
            } else {
                UpdateMutationRate(bestGenome.Fitness);
            }

            FitnessHistory.Add(bestGenome.Fitness);
            liveFitnessPlot.UpdatePlot();

            if(exists) {
                if(bestGenome.Fitness > BestEverGenomeList[existingIndex].Fitness) {
                    BestEverGenomeList[existingIndex] = bestGenome.DeepClone();
                    Console.WriteLine($"[GEN {Generation}] Replaced duplicate genome ID {bestGenome.GenomeID} with better fitness {bestGenome.Fitness:F2}");
                } else {
                    Console.WriteLine($"[GEN {Generation}] Skipped existing genome ID {bestGenome.GenomeID} (lower or equal fitness)");
                }
            } else {
                BestEverGenomeList.Add(bestGenome.DeepClone());
                Console.WriteLine($"[GEN {Generation}] Added new best genome: ID {bestGenome.GenomeID}, score {bestGenome.Game.updateScore}, fitness {bestGenome.Fitness:F2}");
            }

            ConsoleLog(bestGenome);
            generation++;
        }

        private Genome SelectParent(List<Genome> parentGenomes, int tournamentSize = 5) {
            var random = GetRandom();
            var tournament = new List<Genome>(tournamentSize);

            for(int i = 0; i < tournamentSize; i++) {
                tournament.Add(parentGenomes[random.Next(parentGenomes.Count)]);
            }

            return tournament.OrderByDescending(g => g.Fitness).First();
        }

        // ⚡ OPTIMIZATION: Better similarity calculation with early exit potential
        private double Similarity(Genome genomeA, Genome genomeB) {
            var weightsA = genomeA.NeuralNetwork.GetAllWeights();
            var weightsB = genomeB.NeuralNetwork.GetAllWeights();
            var biasesA = genomeA.NeuralNetwork.GetAllBiases();
            var biasesB = genomeB.NeuralNetwork.GetAllBiases();

            if(weightsA.Count != weightsB.Count)
                throw new InvalidOperationException("Genomes have different weight structures.");
            if(biasesA.Count != biasesB.Count)
                throw new InvalidOperationException("Genomes have different bias structures.");

            double diff = 0;

            for(int i = 0; i < weightsA.Count; i++) {
                var matA = weightsA[i];
                var matB = weightsB[i];

                if(matA.RowCount != matB.RowCount || matA.ColumnCount != matB.ColumnCount)
                    throw new InvalidOperationException($"Matrix size mismatch at index {i}");

                diff += (matA - matB).PointwiseAbs().Enumerate().Sum();
            }

            for(int i = 0; i < biasesA.Count; i++) {
                var matA = biasesA[i];
                var matB = biasesB[i];

                if(matA.RowCount != matB.RowCount || matA.ColumnCount != matB.ColumnCount)
                    throw new InvalidOperationException($"Matrix size mismatch at index {i}");

                diff += (matA - matB).PointwiseAbs().Enumerate().Sum();
            }

            return diff;
        }

        private void ApplyFitnessSharing(List<Genome> genomes) {
            const double sharingThreshold = 8.0;

            foreach(var genome in genomes) {
                int similarCount = genomes.Count(o => Similarity(genome, o) < sharingThreshold);
                if(similarCount > 0) {
                    genome.Fitness /= similarCount;
                }
            }
        }

        // ⚡ OPTIMIZATION: Improved crossover with better memory management
        private Genome Crossover(Genome parent1, Genome parent2, double mutationRate, double mutationStrength = 0.1d) {
            var random = GetRandom();
            Genome child = parent1.DeepClone();

            var weightsParent_1 = parent1.NeuralNetwork.GetAllWeights();
            var weightsParent_2 = parent2.NeuralNetwork.GetAllWeights();
            var biasesParent_1 = parent1.NeuralNetwork.GetAllBiases();
            var biasesParent_2 = parent2.NeuralNetwork.GetAllBiases();

            var childWeights = child.NeuralNetwork.GetAllWeights();
            var childBiases = child.NeuralNetwork.GetAllBiases();

            // Weights crossover
            for(int l = 0; l < childWeights.Count; l++) {
                int rows = childWeights[l].RowCount;
                int cols = childWeights[l].ColumnCount;

                for(int r = 0; r < rows; r++) {
                    for(int c = 0; c < cols; c++) {
                        bool takeFromFirst = random.NextDouble() < 0.5;
                        double value = takeFromFirst
                            ? weightsParent_1[l][r, c]
                            : weightsParent_2[l][r, c];
                        childWeights[l][r, c] = value;
                    }
                }
            }

            // Biases crossover
            for(int l = 0; l < childBiases.Count; l++) {
                int rows = childBiases[l].RowCount;
                int cols = childBiases[l].ColumnCount;

                for(int r = 0; r < rows; r++) {
                    for(int c = 0; c < cols; c++) {
                        bool takeFromFirst = random.NextDouble() < 0.5;
                        double value = takeFromFirst
                            ? biasesParent_1[l][r, c]
                            : biasesParent_2[l][r, c];
                        childBiases[l][r, c] = value;
                    }
                }
            }

            child.NeuralNetwork.SetWeights(childWeights);
            child.NeuralNetwork.SetBiases(childBiases);

            if(mutationRate > 0) {
                return child.CloneAndMutate(mutationRate);
            }
            return child;
        }

        void UpdateMutationRate(double currentBestFitness) {
            if(currentBestFitness > bestEverGenome.Fitness + .1) {
                gensSinceImprovement = 0;
                mutationRate = Math.Max(minMutationRate, mutationRate * 0.85);
                previousBestFitness = currentBestFitness;
            } else {
                gensSinceImprovement++;
                Console.WriteLine($"{gensSinceImprovement} Generation Since Improvement");

                if(gensSinceImprovement >= plateauThreshold) {
                    mutationRate = Math.Min(maxMutationRate, mutationRate * 1.15);
                    gensSinceImprovement = 0;
                }
            }
        }

        public void ConsoleLog(Genome smartBestGenome) {
            Console.WriteLine($"Elites preserved: {eliteCount}, Mutated children: {Genomes.Count - eliteCount}");
            Console.WriteLine($"[GENERATION {generation}] |" +
                $"Fitness: {smartBestGenome.Fitness:F10} | " +
                $"Avg Fitness: {AverageFitness:F4} | " +
                $"ID: {smartBestGenome.GenomeID} | " +
                $"Score: {smartBestGenome.Game.updateScore}");

            if(bestEverGenome != null) {
                Console.WriteLine($"Best Ever Genome Till Now ==> " +
                    $"ID: {bestEverGenome.GenomeID} | " +
                    $"Fitness: {bestEverGenome.Fitness:F4} | " +
                    $"Score: {bestEverGenome.Game.updateScore}");
            }

            Console.WriteLine("=_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_=");
            Console.WriteLine();
        }

        public Genome SmartBestGenome => Genomes
            .OrderByDescending(g => g.Score)
            .ThenByDescending(g => g.Fitness)
            .First();

        public double GetMutationRate() => mutationRate;
    }
}