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
        private const int EliteFraction = 5;

        Random random = new Random();

        public List<double> FitnessHistory { get; set; } = [];
        public List<double> SmoothedFitnessHistory { get; private set; } = [];
        public List<double> AverageFitnessHistory { get; } = [];
        public List<Genome> BestEverGenomeList { get; set; } = [];
        public List<Genome> DeadGenomes { get; set; } = [];
        public List<Genome> Genomes { get; set; } = [];

        private int generation = 0;
        private int eliteCount;

        double minMutationRate = 0.005d;
        double mutationRate = 0.015d;
        double maxMutationRate = 0.3d;
        double plateauThreshold = 3;

        int gensSinceImprovement = 0;


        public int Generation {
            get {  return generation; }

            set { generation = value; }
        }
        public int AliveCount => Genomes.Count(g => g.Game.isRunning);

        public double AverageFitness => avgFitness;

        private double avgFitness;

        private Genome bestEverGenome = null!;
        public Genome FinalGenomeFromList { get; set; } = null!;
        public Population(int size, int[] layerSizes) {
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

            List<Genome> UpdateGenomes;

            if (isTraining) {
                UpdateGenomes = Genomes;
            } else {
                UpdateGenomes = BestEverGenomeList;
            }

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
                Vector2 left = new Vector2(-currentDir.Y, currentDir.X);   // rotate left
                Vector2 right = new Vector2(currentDir.Y, -currentDir.X);  // rotate right

                switch(move) {
                case 0: // keep going straight
                    break;
                case 1: // turn left
                    genome.Game.snake.direction = left;
                    break;
                case 2: // turn right
                    genome.Game.snake.direction = right;
                    break;
                }
                genome.Game.Update();

                if(!isGraphShowing)
                    genome.Game.Draw();
            }

            string logText = $"Gen: {Generation} | " +
                             $"Avg: {AverageFitness:F3} | " +
                             $"ALive: {AliveCount} | " +
                             $"Best ID : {SmartBestGenome.GenomeID}";

            Raylib.DrawTextEx(font, logText, new Vector2(380 - offset, offset + cellSize * cellCount + 10), 40, 0, Color.Black);
        }

        public void Evolve() {
            ApplyFitnessSharing(Genomes);

            // Sort by fitness descending
            Genomes = Genomes.OrderByDescending(g => g.Fitness)
                              .ThenByDescending(g => g.Game.updateScore).ToList();

            eliteCount = Math.Max(1, Genomes.Count / EliteFraction);

            var elites = Genomes.Take(eliteCount)
                             .Select(g => g.ShallowClone())
                             .ToList();

            List<Genome> newGen = elites;


            var viableGenomes = Genomes.Where(g => g.Fitness > 20).ToList();
            if(viableGenomes.Count < eliteCount) {
                viableGenomes = Genomes.Take(eliteCount).ToList(); // fallback
            }

            for(int i = 0; i < newGen.Count; i++) {
                newGen[i].AssignID(Generation + 2561, i + 2744);
            }

            // Refill population with mutations
            while(newGen.Count < Genomes.Count) {
                var parent1 = SelectParent(viableGenomes);
                var parent2 = SelectParent(viableGenomes);
                var child = Crossover(parent1, parent2, mutationRate);
                int index = newGen.Count;
                child.AssignID(Generation + 2561, index + 2744);
                newGen.Add(child);
            }
            Genomes = newGen;
        }

        public void EvolveBestEverGenomes() {

            BestEverGenomeList = BestEverGenomeList.OrderByDescending(g => g.Fitness)
                                                   .ThenByDescending(g => g.Game.updateScore)
                                                   .ToList();
            
            // Refill List with mutations
            for(int i = 0; i < BestEverGenomeList.Count; i++) {
                var parent = BestEverGenomeList[i];
                var child = parent.CloneAndMutate(mutationRate);
                BestEverGenomeList[i] = child;
            }
        }

        public void UpdateBestEverGenome(bool isManualUpdate = false) {
            Console.WriteLine($"Trying to add/update best genome...");

            DeadGenomes = DeadGenomes.OrderByDescending(g => g.Game.updateScore)
                                              .ThenByDescending (g => g.Fitness)
                                              .ToList();

            double minFitness = DeadGenomes.Min(g => g.Fitness);
            double shift = (minFitness < 0) ? -minFitness : 0;
            avgFitness = DeadGenomes.Average(g => g.Fitness + shift);
            AverageFitnessHistory.Add(avgFitness);

            Genome bestGenome = DeadGenomes.First();

            DeadGenomes.Clear();
            
            if (isManualUpdate){
                bestGenome.Game.updateScore = bestGenome.Score;
                bestGenome.CalculateFitness();
            }

            // Find existing genome by ID
            int existingIndex = BestEverGenomeList.FindIndex(g => g.GenomeID == bestGenome.GenomeID);
            bool exists = existingIndex != -1;

            // If better fitness (or not present), update or insert
            bool isNewBest = bestEverGenome == null ||
                             bestGenome.Score > bestEverGenome.Score ||
                             (bestGenome.Score == bestEverGenome.Score && bestGenome.Fitness > bestEverGenome.Fitness);

            if(isNewBest) {
                bestEverGenome = bestGenome.DeepClone();
            }

            UpdateMutationRate(bestGenome.Fitness);

            FitnessHistory.Add(bestGenome.Fitness);

            liveFitnessPlot.UpdatePlot();

            if(exists) {
                // Replace if new one is better
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
            var tournament = new List<Genome>();

            for(int i = 0; i < tournamentSize; i++) {
                tournament.Add(parentGenomes[random.Next(parentGenomes.Count)]);
            }

            return tournament.OrderByDescending(g => g.Fitness).First();
        }

        private double Similarity(Genome genomeA, Genome genomeB) {
            var weightsA = genomeA.NeuralNetwork.GetAllWeights();
            var weightsB = genomeB.NeuralNetwork.GetAllWeights();

            var biasesA = genomeA.NeuralNetwork.GetAllBiases();
            var biasesB = genomeB.NeuralNetwork.GetAllBiases();

            if(weightsA.Count != weightsB.Count)
                throw new InvalidOperationException("Genomes have different weight structures.");

            if(biasesA.Count != biasesB.Count)
                throw new InvalidOperationException("Genomes have different biase structures.");

            double diff = 0;
            for(int i = 0; i < weightsA.Count; i++) {
                // Ensure both are MathNet matrices
                var matA = weightsA[i];
                var matB = weightsB[i];

                if(matA.RowCount != matB.RowCount || matA.ColumnCount != matB.ColumnCount)
                    throw new InvalidOperationException($"Matrix size mismatch at index {i}");

                diff += (matA - matB).PointwiseAbs().Enumerate().Sum();
            }

            for(int i = 0; i < biasesA.Count; i++) {
                // Ensure both are MathNet matrices
                var matA = biasesA[i];
                var matB = biasesB[i];

                if(matA.RowCount != matB.RowCount || matA.ColumnCount != matB.ColumnCount)
                    throw new InvalidOperationException($"Matrix size mismatch at index {i}");

                diff += (matA - matB).PointwiseAbs().Enumerate().Sum();
            }

            return diff;
        }

        private void ApplyFitnessSharing(List<Genome> genomes) {
            double sharingThreshold = 5.0;
            foreach(var genome in genomes) {
                int similarCount = genomes.Count(o => Similarity(genome, o) < sharingThreshold);
                genome.Fitness /= similarCount; 
            }
        }

        private Genome Crossover(Genome parent1, Genome parent2, double mutationRate, double mutationStrength = 0.1d) {
            Genome child = parent1.DeepClone(); 

            var weightsParent_1 = parent1.NeuralNetwork.GetAllWeights();
            var weightsParent_2 = parent2.NeuralNetwork.GetAllWeights();

            var biasesParent_1 = parent1.NeuralNetwork.GetAllBiases();
            var biasesParent_2 = parent2.NeuralNetwork.GetAllBiases();

            var childWeights = child.NeuralNetwork.GetAllWeights();
            var childBiases = child.NeuralNetwork.GetAllBiases();

            // Weights crossover + mutation
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

            // Biases crossover + mutation
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
            
            return child.CloneAndMutate(mutationRate);
        }

        void UpdateMutationRate(double currentBestFitness) {
            if(currentBestFitness > bestEverGenome.Fitness) {
                gensSinceImprovement = 0;
                mutationRate = Math.Max(minMutationRate, mutationRate * 0.75f);
            } else {
                gensSinceImprovement++;
                Console.WriteLine($"{gensSinceImprovement} Generation Since Improvement");
                if(gensSinceImprovement >= plateauThreshold) {
                    mutationRate = Math.Min(maxMutationRate, mutationRate * 1.1f);
                    gensSinceImprovement = 0;
                }
            }
        }

        public void ConsoleLog(Genome smartBestGenome) {
            Console.WriteLine($"Elites preserved: {eliteCount}, " +
                $"Mutated children: {Genomes.Count - eliteCount}");
            
            Console.WriteLine($"[GENERATION {generation}] |" +
                $"Fitness: {smartBestGenome.Fitness:F10} | " +
                $"Avg Fitness: {AverageFitness:F4} | " +
                $"ID: {smartBestGenome.GenomeID} | " +
                $"Score: {smartBestGenome.Game.updateScore}");

            if (bestEverGenome != null){
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

        public double GetMutationRate() => 
            mutationRate;
    }
}