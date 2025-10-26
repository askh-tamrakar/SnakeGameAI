using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using static SnakeGameAI.Program;

namespace SnakeGameAI {
    public enum LayerInitType {
        He,
        Xavier
    }

    // ⚡ OPTIMIZATION: Improved matrix stacking performance
    public static class MatrixExtensions {
        public static Matrix<double> Stack(this Matrix<double> rowVector, int times) {
            // Faster implementation using preallocated array and SetRow
            var result = Matrix<double>.Build.Dense(times, rowVector.ColumnCount);
            var rowArray = rowVector.Row(0).ToArray();

            for(int i = 0; i < times; i++) {
                result.SetRow(i, rowArray);
            }
            return result;
        }
    }

    // Neural Network class - Optimized
    public class NeuralNetwork {
        public static List<Matrix<double>> CumulativeMutations = new();
        public List<LayerDense> Layers = new();
        public List<object> Activations = new();
        private readonly int outputSize;

        // ⚡ OPTIMIZATION: Thread-local Random for better performance in parallel scenarios
        [ThreadStatic]
        private static Random? _threadRandom;
        private static Random ThreadRandom => _threadRandom ??= new Random(Guid.NewGuid().GetHashCode());

        public NeuralNetwork(int[] layerSizes, LayerInitType initType = LayerInitType.He) {
            for(int i = 0; i < layerSizes.Length - 1; i++) {
                int inputSize = layerSizes[i];
                int outputSizeLayer = layerSizes[i + 1];
                var dense = new LayerDense(inputSize, outputSizeLayer,
                    i == layerSizes.Length - 2 ? LayerInitType.Xavier : initType);
                Layers.Add(dense);

                object activation = (i == layerSizes.Length - 2) ?
                    new ActivationSoftmax() : new ActivationRelu();
                Activations.Add(activation);
            }

            this.outputSize = layerSizes.Last();

            foreach(var layer in Layers) {
                CumulativeMutations.Add(Matrix<double>.Build.Dense(layer.Weights.RowCount, layer.Weights.ColumnCount));
                CumulativeMutations.Add(Matrix<double>.Build.Dense(layer.Biases.RowCount, layer.Biases.ColumnCount));
            }
        }

        public (List<double>, int) Predict(Matrix<double> input) {
            Matrix<double> output = input;

            for(int i = 0; i < Layers.Count; i++) {
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

        // ⚡ OPTIMIZATION: Better memory access patterns, using ThreadRandom
        public void Mutate(double mutationRate) {
            var rand = ThreadRandom;

            void MutateMatrix(Matrix<double> matrix, Matrix<double> mutationCounter) {
                int rows = matrix.RowCount;
                int cols = matrix.ColumnCount;

                // Better cache locality with row-major access
                for(int i = 0; i < rows; i++) {
                    for(int j = 0; j < cols; j++) {
                        if(rand.NextDouble() < mutationRate) {
                            double mutationValue;
                            double adaptiveFactor = 1.0 / (1.0 + population.Generation / 100.0);

                            if(rand.NextDouble() < 0.90) {
                                mutationValue = Normal.Sample(0, adaptiveFactor);
                            } else {
                                mutationValue = Cauchy.Sample(0, 0.5 * adaptiveFactor);
                            }

                            matrix[i, j] += mutationValue;
                            mutationCounter[i, j] = Math.Min(mutationCounter[i, j] + 0.1, 255);
                        }
                    }
                }
            }

            for(int i = 0; i < Layers.Count; i++) {
                var layer = Layers[i];
                var cmWeights = CumulativeMutations[i * 2];
                var cmBiases = CumulativeMutations[i * 2 + 1];

                MutateMatrix(layer.Weights, cmWeights);
                MutateMatrix(layer.Biases, cmBiases);

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

        // Forward pass with optimized matrix stacking
        public void Forward(Matrix<double> inputs) {
            var biasesRepeated = Biases.Stack(inputs.RowCount);
            Output = (inputs * Weights) + biasesRepeated;
        }
    }

    // ReLU Activation - Optimized with Map
    public class ActivationRelu {
        public Matrix<double> Output { get; set; } = null!;

        public void Forward(Matrix<double> inputs) {
            Output = inputs.Map(x => Math.Max(0, x));
        }
    }

    // ⚡ OPTIMIZATION: Streamlined Softmax to avoid intermediate allocations
    public class ActivationSoftmax {
        public Matrix<double> Output = null!;

        public void Forward(Matrix<double> inputs) {
            int rows = inputs.RowCount;
            int cols = inputs.ColumnCount;

            var expValues = Matrix<double>.Build.Dense(rows, cols);

            // Calculate exp values with numerical stability
            for(int i = 0; i < rows; i++) {
                var row = inputs.Row(i);
                double max = row.Maximum();
                var exp = row.Subtract(max).Map(Math.Exp);
                expValues.SetRow(i, exp);
            }

            // Normalize directly without intermediate matrices
            var sums = expValues.RowSums();
            Output = Matrix<double>.Build.Dense(rows, cols);

            for(int i = 0; i < rows; i++) {
                for(int j = 0; j < cols; j++) {
                    Output[i, j] = expValues[i, j] / sums[i];
                }
            }
        }
    }

    public static class WeightInitializer {
        [ThreadStatic]
        private static Random? _random;
        private static Random GetRandom() => _random ??= new Random(Guid.NewGuid().GetHashCode());

        public static Matrix<double> HeInitialization(int inputSize, int outputSize) {
            double stdDev = Math.Sqrt(2.0 / inputSize);
            var normal = new Normal(0, stdDev);
            return Matrix<double>.Build.Random(inputSize, outputSize, normal);
        }

        public static Matrix<double> XavierInitialization(int inputSize, int outputSize) {
            double limit = Math.Sqrt(6.0 / (inputSize + outputSize));
            var uniform = new ContinuousUniform(-limit, limit);
            return Matrix<double>.Build.Random(inputSize, outputSize, uniform);
        }
    }
}