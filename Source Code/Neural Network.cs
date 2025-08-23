using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using static SnakeGameAI.Program;

namespace SnakeGameAI {
    public enum LayerInitType {
        He,
        Xavier
    }

    // Utility extension for repeating rows
    public static class MatrixExtensions {
        public static Matrix<double> Stack(this Matrix<double> rowVector, int times) {
            return Matrix<double>.Build.Dense(times, rowVector.ColumnCount, (i, j) => rowVector[0, j]);
        }
    }

    // Neural Network class
    public class NeuralNetwork {
        public static List<Matrix<double>> CumulativeMutations = new();
        public List<LayerDense> Layers = new();
        public List<object> Activations = new();
        private int outputSize;

        public NeuralNetwork(int[] layerSizes, LayerInitType initType = LayerInitType.He) {
            for(int i = 0; i < layerSizes.Length - 1; i++) {
                int inputSize = layerSizes[i];
                int outputSize = layerSizes[i + 1];

                var dense = new LayerDense(inputSize, outputSize, i == layerSizes.Length - 2 ? LayerInitType.Xavier : initType);
                Layers.Add(dense);

                object activation = (i == layerSizes.Length - 2) ? new ActivationSoftmax() : new ActivationRelu();
                Activations.Add(activation);
            }

            this.outputSize = layerSizes.Last();

            foreach(var layer in Layers) {
                // Add mutation matrix for weights
                CumulativeMutations.Add(Matrix<double>.Build.Dense(layer.Weights.RowCount, layer.Weights.ColumnCount));

                // Add mutation matrix for biases
                CumulativeMutations.Add(Matrix<double>.Build.Dense(layer.Biases.RowCount, layer.Biases.ColumnCount));
            }
        }

        public (List<double>, int) Predict(Matrix<double> input) {
            Matrix<double> output = input;

            for(int i = 0;i < Layers.Count;i++) {
                Layers[i].Forward(output);

                if(Activations[i] is ActivationRelu relu)
                    relu.Forward(Layers[i].Output);
                else if(Activations[i] is ActivationSoftmax softmax)
                    softmax.Forward(Layers[i].Output);

                output = (Activations[i] as dynamic).Output;
            }

            var outputRow = output.Row(0);
            return (outputRow.ToList(), outputRow.MaximumIndex());
        }

        public List<Matrix<double>> GetAllWeights() => Layers.Select(l => l.Weights.Clone()).ToList();
        public List<Matrix<double>> GetAllBiases() => Layers.Select(l => l.Biases.Clone()).ToList();

        public void SetWeights(List<Matrix<double>> newWeights) {
            if(newWeights.Count != Layers.Count)
                throw new ArgumentException("Weights count mismatch.");

            for(int i = 0; i < Layers.Count; i++) {
                if(newWeights[i].RowCount != Layers[i].Weights.RowCount ||
                    newWeights[i].ColumnCount != Layers[i].Weights.ColumnCount) {
                    throw new ArgumentException($"Weight matrix size mismatch at layer {i}.");
                }

                Layers[i].Weights = newWeights[i].Clone();
            }
        }

        public void SetBiases(List<Matrix<double>> newBiases) {
            if(newBiases.Count != Layers.Count)
                throw new ArgumentException("Biases count mismatch.");

            for(int i = 0; i < Layers.Count; i++) {
                if(newBiases[i].RowCount != Layers[i].Biases.RowCount ||
                    newBiases[i].ColumnCount != Layers[i].Biases.ColumnCount) {
                    throw new ArgumentException($"Bias matrix size mismatch at layer {i}.");
                }

                Layers[i].Biases = newBiases[i].Clone();
            }
        }

        public NeuralNetwork Clone() {
            var clone = new NeuralNetwork(Layers.Select(l => l.InputsCount).Append(outputSize).ToArray());

            for(int i = 0; i < Layers.Count; i++) {
                clone.Layers[i].Weights = Layers[i].Weights.Clone();
                clone.Layers[i].Biases = Layers[i].Biases.Clone();
            }

            return clone;
        }

        public void Mutate(double mutationRate) {
            var rand = new Random();

            void MutateMatrix(ref Matrix<double> matrix, ref Matrix<double> mutationCounter) {
                for(int i = 0;i < matrix.RowCount;i++) {
                    for(int j = 0;j < matrix.ColumnCount;j++) {
                        if(rand.NextDouble() < mutationRate) {
                            if(rand.NextDouble() < 0.95){
                                double freq = population.Generation % 100;
                                double dynamicNoise = Math.Sin(freq * Math.PI / 100.0) * 0.05;
                                matrix[i, j] += Normal.Sample(0, dynamicNoise); // 95% small mutations 
                            }      
                            else
                                matrix[i, j] = Cauchy.Sample(0, 1.0);         // 5% full reset
                            mutationCounter[i, j] = Math.Min(mutationCounter[i, j] + 0.1, 255);
                        }
                    }
                }
            }

            
            for(int i = 0; i < Layers.Count; i++) {
                var layer = Layers[i];
                var cmWeights = CumulativeMutations[i * 2];
                var cmBiases = CumulativeMutations[i * 2 + 1];

                MutateMatrix(ref layer.Weights, ref cmWeights);
                MutateMatrix(ref layer.Biases, ref cmBiases);

                CumulativeMutations[i * 2] = cmWeights;
                CumulativeMutations[i * 2 + 1] = cmBiases;
            }
        }

        public List<LayerDense> GetLayers() => Layers;
        public List<object> GetActivations() => Activations;
        public LayerDense GetLayer(int index) => Layers[index];

    }


    // Dense Layer
    public class LayerDense {
        public int InputsCount { get; }
        public int NeuronsCount { get; }

        public Matrix<double> Weights;
        public Matrix<double> Biases;
        public Matrix<double> Output { get; set; } = null!;

        public LayerDense(int inputsCount, int neuronsCount, LayerInitType initType) {
            InputsCount = inputsCount;
            NeuronsCount = neuronsCount;

            if(initType == LayerInitType.He) {
                Weights = WeightInitializer.HeInitialization(inputsCount, neuronsCount);
            } else {
                Weights = WeightInitializer.XavierInitialization(inputsCount, neuronsCount);
            }
            var initialBiases = new ContinuousUniform(-0.1, 0.1);
            Biases = Matrix<double>.Build.Random(1, neuronsCount, initialBiases); 
        }

        public void Forward(Matrix<double> inputs) {
            var biasesRepeated = Biases.Stack(inputs.RowCount);
            Output = (inputs * Weights) + biasesRepeated;
        }
    }

    // ReLU Activation
    public class ActivationRelu {
        public Matrix<double> Output { get; set; } = null!;
        public void Forward(Matrix<double> inputs) {
            Output = inputs.Map(x => Math.Max(0, x));
        }
    }

    // Softmax Activation
    public class ActivationSoftmax {
        public Matrix<double> Output = null!;

        public void Forward(Matrix<double> inputs) {
            int rows = inputs.RowCount;
            int cols = inputs.ColumnCount;
            var expValues = Matrix<double>.Build.Dense(rows, cols);

            for(int i = 0; i < rows; i++) {
                var row = inputs.Row(i);
                double max = row.Maximum();
                var exp = row.Subtract(max).Map(Math.Exp);
                expValues.SetRow(i, exp);
            }

            var sums = expValues.RowSums().ToColumnMatrix();
            var repeatedSums = Matrix<double>.Build.Dense(expValues.RowCount, expValues.ColumnCount, (i, j) => sums[i, 0]);
            Output = expValues.PointwiseDivide(repeatedSums);
        }
    }

    public static class WeightInitializer {
        private static readonly Random random = new();

        // He Initialization for ReLU
        public static Matrix<double> HeInitialization(int inputSize, int outputSize) {
            double stdDev = Math.Sqrt(2.0 / inputSize);
            var normal = new Normal(0, stdDev);
            return Matrix<double>.Build.Random(inputSize, outputSize, normal);
        }

        // Xavier Uniform for Softmax/tanh
        public static Matrix<double> XavierInitialization(int inputSize, int outputSize) {
            double limit = Math.Sqrt(6.0 / (inputSize + outputSize));
            var uniform = new ContinuousUniform(-limit, limit);
            return Matrix<double>.Build.Random(inputSize, outputSize, uniform);
        }
        private static double RandomDouble(double min, double max) {
            return min + (random.NextDouble() * (max - min));
        }

        private static double GaussianRandom(double mean, double stdDev) {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                                   Math.Sin(2.0 * Math.PI * u2);
            return mean + stdDev * randStdNormal;
        }
    }
}