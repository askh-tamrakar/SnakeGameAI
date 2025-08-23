using MathNet.Numerics.LinearAlgebra;
using Raylib_cs;
using ScottPlot;
using ScottPlot.Plottables;
using ScottPlot.WinForms;
using System.Drawing.Printing;
using System.Numerics;
using System.Text.Json;
using static SnakeGameAI.Program;
using static SnakeGameAI.Snake_Game;
using Font = Raylib_cs.Font;
using RayColor = Raylib_cs.Color;
using Rectangle = Raylib_cs.Rectangle;
using ScottColor = ScottPlot.Color;

namespace SnakeGameAI {

    public class AI_Debugger {

        public class NeuralNetDisplaySettings {
            public float LayerSpacing { get; set; } = 200f;
            public float NeuronSpacing { get; set; } = 10f;
            public float ZoomOverride { get; set; } = 1f;
            public float NodeRadius { get; set; } = 10f;
            public float FontSize { get; set; } = 18f;
        }

        public NeuralNetDisplaySettings displaySettings = new();
        public static Font font = new();

        string settingsPath = "settings.json";

        public bool isLogScale = false;
        public float graphZoom = 1.0f;
        public Vector2 GraphOffset = new(0, 0);

        const float LayerMin = 20f, LayerMax = 400f;
        const float NeuronMin = 2f, NeuronMax = 100f;
        const float ZoomMin = 0.1f, ZoomMax = 3f;
        const float RadiusMin = 5f, RadiusMax = 30f;
        const float FontMin = 10f, FontMax = 40f;

        float Lerp(float a, float b, float t) => a + (b - a) * t;

        public Vector2 CatmullRom(Vector2 p0, Vector2 p1, Vector2 p2, Vector2 p3, float t) {
            float t2 = t * t;
            float t3 = t2 * t;
            return 0.5f * ((2 * p1) +
                           (-p0 + p2) * t +
                           (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                           (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
        }

        public void SaveSettings() {
            var options = new JsonSerializerOptions {
                WriteIndented = true,
                IncludeFields = true
            };
            string json = JsonSerializer.Serialize(displaySettings, options);
            File.WriteAllText(settingsPath, json);
        }

        public void LoadSettings() {
            if(File.Exists(settingsPath)) {
                string json = File.ReadAllText(settingsPath);
                displaySettings = JsonSerializer.Deserialize<NeuralNetDisplaySettings>(json) ?? new();
            }
        }

        public void DrawFitnessGraph(Population population) {

            int graphX = 0;
            int graphY = 0;
            int graphWidth = (int)windowDim.X;
            int graphHeight = (int)windowDim.Y;
            int margin = 50;
            int borderSize = 5;

            int axisX = graphX + margin;
            int axisY = graphY + graphHeight - margin;

            float panX = 0;
            float panY = 0;
            float zoomFactor = 1.0f;
            bool isDragging = false;

            Vector2 lastMousePos = Raylib.GetMousePosition();

            Raylib.DrawTextEx(font, $"Gen -->", new Vector2(graphX + (int)(0.5 * (graphWidth - 2 * margin)), axisY + 20), 18, 0, RayColor.RayWhite);
            Raylib.DrawTextEx(font, "Fitness over Generations", new Vector2(axisX, graphY - 30), 40, 0, RayColor.DarkGreen);

            // Draw border
            Rectangle rect = new Rectangle(graphX, graphY, graphWidth, graphHeight);
            Raylib.DrawRectangle((int)rect.X + borderSize, (int)rect.Y + borderSize, (int)rect.Width - borderSize, (int)rect.Height - borderSize, RayColor.RayWhite);
            Raylib.DrawRectangleLinesEx(rect, borderSize, RayColor.Red);

            // Draw axis lines
            Raylib.DrawLine(axisX, graphY + margin, axisX, axisY, RayColor.Black); // Y-axis
            Raylib.DrawLine(axisX, axisY, graphX + graphWidth - margin, axisY, RayColor.Black); // X-axis

            var history = population.FitnessHistory;
            if(history.Count < 2)
                return;

            double maxFitness = history.Max();
            double minFitness = history.Min();
            int count = history.Count;
            

            double norm(double v) => isLogScale ? (Math.Log10(v + 1e-5) - Math.Log10(minFitness + 1e-5)) /
                                          (Math.Log10(maxFitness + 1e-5) - Math.Log10(minFitness + 1e-5))
                                        : (v - minFitness) / (maxFitness - minFitness);

            // X-axis: ticks and labels (generations)
            int xTicks = Math.Min(count, 10);
            for(int i = 0; i <= xTicks; i++) {
                float t = i / (float)xTicks;
                int x = axisX + (int)(t * (graphWidth - 2 * margin));
                Raylib.DrawLine(x, axisY - 5, x, axisY + 5, RayColor.Black);

                int gen = (int)(t * count);
                string gen_text = $"{gen}";
                Raylib.DrawTextEx(font, gen_text, new Vector2(x - 10, axisY + 8), 15, 0, RayColor.Black);
            }

            // Y-axis: ticks and labels (fitness)
            int yTicks = 20;
            for(int i = 0; i <= yTicks; i++) {
                float t = i / (float)yTicks;
                int y = axisY - (int)(t * (graphHeight - 2 * margin));
                Raylib.DrawLine(axisX - 5, y, axisX + 5, y, RayColor.Black);

                double val = isLogScale
                    ? Math.Pow(10,
                        Math.Log10(minFitness + 1e-5) +
                        t * (Math.Log10(maxFitness + 1e-5) - Math.Log10(minFitness + 1e-5)))
                    : minFitness + t * (maxFitness - minFitness);

                Raylib.DrawTextEx(font, $"{val:F3}", new Vector2(axisX - 50, y - 8), 15, 0, RayColor.Black);
            }

            // Plot line
            for(int i = 1; i < count; i++) {
                float x1 = axisX + ((i - 1) / (float)(count - 1)) * (graphWidth - 2 * margin);
                float y1 = axisY - (float)(norm(history[i - 1]) * (graphHeight - 2 * margin));

                float x2 = axisX + (i / (float)(count - 1)) * (graphWidth - 2 * margin);
                float y2 = axisY - (float)(norm(history[i]) * (graphHeight - 2 * margin));
                Raylib.DrawLineEx(new Vector2(x1, y1), new Vector2(x2, y2), 2.5f, RayColor.Blue); // Bold line
            }

            //Mark the bestFitness
            int bestIndex = history.IndexOf(maxFitness);
            float bestX = axisX + (bestIndex / (float)(count - 1)) * (graphWidth - 2 * margin);
            float bestY = axisY - (float)((maxFitness - minFitness) / (maxFitness - minFitness) * (graphHeight - 2 * margin));

            // Draw cross (two intersecting lines)
            int crossSize = 6;
            float lineThickness = 2.5f;
            RayColor lineColor = AppColors.YellowHighlight.Ray;
            Raylib.DrawLineEx(new Vector2((int)(bestX - crossSize), (int)(bestY - crossSize)),
                              new Vector2((int)(bestX + crossSize), (int)(bestY + crossSize)),
                              lineThickness, lineColor);

            Raylib.DrawLineEx(new Vector2((int)(bestX - crossSize), (int)(bestY + crossSize)),
                              new Vector2((int)(bestX + crossSize), (int)(bestY - crossSize)), 
                              lineThickness, lineColor);

            //Raylib.DrawCircle((int)bestX, (int)bestY, 5, Color.Yellow);


            if(Raylib.IsMouseButtonPressed(MouseButton.Middle)) {
                isDragging = true;
                lastMousePos = Raylib.GetMousePosition();
            }

            if(Raylib.IsMouseButtonReleased(MouseButton.Middle)) {
                isDragging = false;
            }

            if(isDragging) {
                Vector2 delta = Raylib.GetMousePosition() - lastMousePos;
                panX += delta.X;
                panY += delta.Y;
                lastMousePos = Raylib.GetMousePosition();
            }

            float wheel = Raylib.GetMouseWheelMove();
            if(Raylib.CheckCollisionPointRec(Raylib.GetMousePosition(), new Rectangle(graphX, graphY, graphWidth, graphHeight)))
                zoomFactor = Math.Clamp(zoomFactor + wheel * 0.1f, 0.5f, 3.0f);

        }

        void DrawSlider(ref float value, float min, float max, int x, int y, string label, RayColor color) {
            int width = 200;
            Raylib.DrawText(label, x, y - 20, 20, RayColor.LightGray);
            Raylib.DrawRectangle(x, y, width, 10, RayColor.DarkGray);

            Vector2 mouse = Raylib.GetMousePosition();
            Rectangle barRect = new(x, y - 5, width, 20);

            if(Raylib.IsMouseButtonDown(MouseButton.Left) && Raylib.CheckCollisionPointRec(mouse, barRect)) {
                float t = Math.Clamp((mouse.X - x) / width, 0f, 1f);
                float target = min + t * (max - min);
                value = value * 0.85f + target * 0.15f; 
            }

            // Draw draggable knob
            float knobX = x + (value - min) / (max - min) * width;
            Raylib.DrawRectangle((int)knobX - 5, y - 5, 10, 20, color);

            // Value label
            Raylib.DrawText($"{value:F1}", x + width + 15, y - 4, 18, color);
        }

        public void DrawSlider() {
            int x = 1600;
            int y = 780;
            int spacing = 40;

            float LayerSpacing = displaySettings.LayerSpacing;
            DrawSlider(ref LayerSpacing, LayerMin, LayerMax, x, y, "Layer Spacing", RayColor.Red);
            displaySettings.LayerSpacing = LayerSpacing;
            y += spacing;

            float NeuronSpacing = displaySettings.NeuronSpacing;
            DrawSlider(ref NeuronSpacing, NeuronMin, NeuronMax, x, y, "Neuron Spacing", RayColor.Blue);
            displaySettings.NeuronSpacing = NeuronSpacing;
            y += spacing;

            float ZoomOverride = displaySettings.ZoomOverride;
            DrawSlider(ref ZoomOverride, ZoomMin, ZoomMax, x, y, "Zoom", RayColor.Green);
            displaySettings.ZoomOverride = ZoomOverride;
            y += spacing;

            float NodeRadius = displaySettings.NodeRadius;
            DrawSlider(ref NodeRadius, RadiusMin, RadiusMax, x, y, "Node Radius", RayColor.Purple);
            displaySettings.NodeRadius = NodeRadius;
            y += spacing;

            float FontSize = displaySettings.FontSize;
            DrawSlider(ref FontSize, FontMin, FontMax, x, y, "Font Size", RayColor.Orange);
            displaySettings.FontSize = FontSize;
        }

        public void DrawDebugPanel(Genome genome, Population population, Vector2 panelOrigin,
            string genomeID,
            List<Genome>? bestGenomesEver = null ) {
            int lineHeight = 24;
            int fontSize = 28;
            int stepsLeft, maxSteps;
            int y = (int)panelOrigin.Y;
            int x = (int)panelOrigin.X;

            maxSteps = genome.Game.maxStepsWithoutFood + (genome.Score * 8);
            stepsLeft = maxSteps - genome.Game.stepsSinceLastFood;

            Raylib.DrawTextEx(font, "Genome Debug Panel", new Vector2(x, y), 36, 0, RayColor.DarkBlue);
            y += lineHeight * 2;

            // Generation
            Raylib.DrawTextEx(font, $"Generation: {population.Generation}", new Vector2(x, y), fontSize, 0, RayColor.Brown);
            y += lineHeight;

            Raylib.DrawTextEx(font, $"Mutation Rate: {population.GetMutationRate():F6}", new Vector2(x, y), fontSize, 0, RayColor.DarkBrown);
            y += lineHeight;

            Raylib.DrawTextEx(font, $"Genome Index: #{genomeID}", new Vector2(x, y), fontSize, 0, RayColor.DarkBrown);
            y += lineHeight;

            // Fitness, Score, Steps
            Raylib.DrawTextEx(font, $"Fitness: {genome.Fitness:F2}", new Vector2(x, y), fontSize, 0, RayColor.DarkGray);
            y += lineHeight;
            Raylib.DrawTextEx(font, $"Score: {genome.Score}", new Vector2(x, y), fontSize, 0, RayColor.DarkGray);
            y += lineHeight;
            Raylib.DrawTextEx(font, $"Steps Survived: {genome.Game.stepsSurvived}", new Vector2(x, y), fontSize, 0, RayColor.DarkGray);
            y += lineHeight;
            Raylib.DrawTextEx(font, $"Maximum Steps : {maxSteps}", new Vector2(x, y), fontSize, 0, RayColor.DarkGray);
            y += lineHeight;
            Raylib.DrawTextEx(font, $"Steps Left: {stepsLeft}", new Vector2(x, y), fontSize, 0, RayColor.DarkGray);
            y += lineHeight;
            Raylib.DrawTextEx(font, $"Length: {genome.SnakeLength}", new Vector2(x, y), fontSize, 0, RayColor.DarkGray);
            y += lineHeight;

            // Direction
            Raylib.DrawTextEx(font, $"Direction: {genome.Game.snake.direction}", new Vector2(x, y), fontSize, 0, RayColor.DarkGray);
            y += lineHeight;

            // Best-Ever status if list is provided
            if(bestGenomesEver != null) {
                int bestIndex = bestGenomesEver.FindIndex(g => g == genome);
                if(bestIndex != -1) {
                    Raylib.DrawTextEx(font, $"This genome index #{bestIndex}", new Vector2(x, y), fontSize, 0, RayColor.Blue);
                } else {
                    Raylib.DrawTextEx(font, "Not in BestEver List", new Vector2(x, y), fontSize, 0, RayColor.Gray);
                }
                y += lineHeight;
                Raylib.DrawTextEx(font, $"BestEver List Count: {bestGenomesEver.Count}", new Vector2(x, y), fontSize, 0, RayColor.Blue);
                y += lineHeight;
            }

            // Neural Network output
            y += lineHeight;
            Raylib.DrawTextEx(font, "Raw NN Output:", new Vector2(x, y), fontSize, 0, RayColor.Maroon);
            y += lineHeight;

            var input = Matrix<double>.Build.Dense(1, genome.Game.GetInputs().Length, (i, j) => genome.Game.GetInputs()[j]);
            var (prediction, _) = genome.NeuralNetwork.Predict(input);

            for(int i = 0; i < prediction.Count; i++) {
                string label = i switch {
                    0 => "Forward",
                    1 => "Left",
                    2 => "Right",
                    _ => $"Out {i}"
                };
                Raylib.DrawTextEx(font, $"{label}: {prediction[i]:F2}", new Vector2(x, y), fontSize, 0, RayColor.DarkGreen);
                y += lineHeight;
            }

        }

        public void DrawNeuralNetwork(Genome genome, Vector2 origin, int width, int height) {

            RayColor darkGreen = new RayColor(43, 51, 24, 255);
            RayColor nodeColor = AppColors.ForestGreen.Ray;
            RayColor connectionColor = RayColor.Gray;

            var input = genome.Game.GetInputs();
            Matrix<double> inputMatrix = Matrix<double>.Build.Dense(1, input.Length, (i, j) => input[j]);

            var (prediction, _) = genome.NeuralNetwork.Predict(inputMatrix);
            Matrix<double> predictionMatrix = Matrix<double>.Build.Dense(1, prediction.Count, (i, j) => prediction[j]);

            List<int> layerDense = [
                genome.NeuralNetwork.GetLayers()[0].InputsCount
            ];

            for(int i = 0; i < genome.NeuralNetwork.GetLayers().Count ; i++) {
                layerDense.Add(genome.NeuralNetwork.GetLayers()[i].NeuronsCount);
            }

            List<Matrix<double>> OutputOfEachNeuron = [
                inputMatrix,
            ];
            for(int i = 0; i < genome.NeuralNetwork.Activations.Count; i++) {

                if(genome.NeuralNetwork.Activations[i] is ActivationRelu relu)
                    relu.Forward(genome.NeuralNetwork.GetLayers()[i].Output);
                var output = (genome.NeuralNetwork.Activations[i] as dynamic).Output;
                OutputOfEachNeuron.Add(output);
            }

            OutputOfEachNeuron.Add(predictionMatrix);


            float zoom = displaySettings.ZoomOverride;

            float layerSpacing = displaySettings.LayerSpacing * displaySettings.ZoomOverride;
            float scaledNeuronSpacing = displaySettings.NeuronSpacing * displaySettings.ZoomOverride;
            float scaledRadius = displaySettings.NodeRadius * displaySettings.ZoomOverride;


            List<Vector2[]> positions = new();

            // Scaling According to Width
            float networkTotalWidth = (layerDense.Count - 1) * layerSpacing;
            float startX = origin.X + (width - networkTotalWidth) / 2f;

            // Calculate positions of all neurons
            for(int l = 0; l < layerDense.Count; l++) {
                Vector2[] layerPositions = new Vector2[layerDense[l]];

                //Scaling According to Height
                float layerHeight = layerDense[l] * 2 * scaledRadius + (layerDense[l] - 1) * scaledNeuronSpacing;
                float startY = origin.Y + (height - layerHeight) / 2f;

                for(int n = 0; n < layerDense[l]; n++) {
                    float x = startX + l * layerSpacing;
                    float y = startY + n * (2 * scaledRadius + scaledNeuronSpacing);

                    layerPositions[n] = new Vector2(x, y);
                }

                positions.Add(layerPositions);
            }

            // Draw connections first
            for(int l = 0; l < layerDense.Count - 1; l++) {
                Vector2[] currentLayer = positions[l];
                Vector2[] nextLayer = positions[l + 1];

                // Select correct weight matrix
                Matrix<double> weights = l == 0 ? genome.NeuralNetwork.GetAllWeights()[0] :
                                         l == 1 ? genome.NeuralNetwork.GetAllWeights()[1] :
                                         l == 2 ? genome.NeuralNetwork.GetAllWeights()[2] :
                                         l == 3 ? genome.NeuralNetwork.GetAllWeights()[3] :
                                                  Matrix<double>.Build.Dense(0, 0); // Empty fallback

                for(int i = 0; i < weights.RowCount; i++) {
                    for(int j = 0; j < weights.ColumnCount; j++) {
                        if(i >= currentLayer.Length || j >= nextLayer.Length)
                            continue;

                        float weight = (float)weights[i, j];
                        float thickness = MathF.Min(4f, MathF.Max(1f, MathF.Abs(weight) * 3f * zoom));
                        RayColor color = weight > 0 ? RayColor.Red : RayColor.Blue;

                        Raylib.DrawLineEx(currentLayer[i], nextLayer[j], thickness, color);
                    }
                }
            }

            // Draw neurons and labels after connections
            for(int l = 0; l < layerDense.Count; l++) {
                for(int n = 0; n < layerDense[l]; n++) {
                    Vector2 pos = positions[l][n];

                    // draw neuron
                    Raylib.DrawCircle((int)pos.X, (int)pos.Y, scaledRadius, nodeColor);

                    // Write output text
                    Vector2 posOutput = new(pos.X - (float)(scaledRadius * 0.63066), pos.Y - (scaledRadius / 2) - (float)0.05); 
                    
                    Matrix<double> outputMatrix = OutputOfEachNeuron[l];
                    double outputOfNeuron = outputMatrix[0, n];

                    Raylib.DrawTextEx(font, $"{outputOfNeuron:F2}", posOutput, scaledRadius, 0, AppColors.YellowHighlight.Ray);

                    // Draw layer label only for top neuron
                    if(n == 0) {
                        string label = l switch {
                            0 => "Input",
                            1 => "Hidden 1",
                            2 => "Hidden 2",
                            3 => "Hidden 3",
                            4 => "Output",
                            _ => $"L{l}"
                        };
                        int fontSize = (int)(displaySettings.FontSize * zoom);
                        Raylib.DrawText(label, (int)(pos.X - 30 * zoom), (int)(pos.Y - 40 * zoom), fontSize, AppColors.JungleGreen.Ray);
                    }
                }
            }

            Raylib.DrawText("Neural Network", (int)origin.X + 5, (int)origin.Y - 40, 28, RayColor.Black);
        }

        public static void DrawHeatmap(Matrix<double> matrix, int startX, int startY, int cellSize, double maxMutationCount = -1) {
            int rows = matrix.RowCount;
            int cols = matrix.ColumnCount;

            // Automatically find the max value for normalization if not provided
            if(maxMutationCount <= 0) {
                maxMutationCount = matrix.Enumerate().Max();
            }

            for(int i = 0;i < rows;i++) {
                for(int j = 0;j < cols;j++) {
                    double value = matrix[i, j];
                    float intensity = (float)(value / maxMutationCount);  // Normalize 0 to 1
                    intensity = Math.Clamp(intensity, 0f, 1f);

                    RayColor color = new RayColor(
                        (byte)(intensity * 255),
                        (byte)0,
                        (byte)((1 - intensity) * 255),
                        (byte)255
                    );

                    Raylib.DrawRectangle(startX + j * cellSize, startY + i * cellSize, cellSize, cellSize, color);
                }
            }
        }

    }

    public class LiveFitnessPlot : IDisposable {
        public FormsPlot formsPlot = null!;

        private Scatter scatterRaw = null!;
        private Scatter scatterAverage = null!;
        private Scatter scatterSmoothed = null!;

        private readonly Thread uiThread;
        private readonly Population population;
        private readonly int smoothingWindow;
        private bool disposedValue;

        // ctor: supply the population and desired smoothing window (e.g. 5)
        public LiveFitnessPlot(Population population, int smoothingWindow = 5) {
            this.population = population ?? throw new ArgumentNullException(nameof(population));
            this.smoothingWindow = Math.Max(1, smoothingWindow);

            uiThread = new Thread(() => {
                var form = new Form {
                    Text = "Live Fitness Plot",
                    Width = 900,
                    Height = 650
                };

                formsPlot = new FormsPlot { Dock = DockStyle.Fill };
                form.Controls.Add(formsPlot);

                // initial empty data
                double[] value = [0];
                double[] xs = value;
                double[] ys = value;
                double[] ysAvg = value;
                double[] ysSmooth = value;

                scatterRaw = formsPlot.Plot.Add.Scatter(xs, ys);

                scatterAverage = formsPlot.Plot.Add.Scatter(xs, ysAvg);

                scatterSmoothed = formsPlot.Plot.Add.Scatter(xs, ysSmooth);

                formsPlot.Plot.Title("Fitness Over Generations");
                formsPlot.Plot.XLabel("Generation");
                formsPlot.Plot.YLabel("Fitness");

                // change figure colors
                formsPlot.Plot.FigureBackground.Color = ScottColor.FromHex("#181818");
                formsPlot.Plot.DataBackground.Color = ScottColor.FromHex("#1f1f1f");

                // change axis and grid colors
                formsPlot.Plot.Axes.Color(ScottColor.FromHex("#d7d7d7"));
                formsPlot.Plot.Grid.MajorLineColor = ScottColor.FromHex("#404040");

                // change legend colors
                formsPlot.Plot.Legend.BackgroundColor = ScottColor.FromHex("#404040");
                formsPlot.Plot.Legend.FontColor = ScottColor.FromHex("#d7d7d7");
                formsPlot.Plot.Legend.OutlineColor = ScottColor.FromHex("#d7d7d7");

                formsPlot.Refresh();

                Application.Run(form);
            });

            uiThread.SetApartmentState(ApartmentState.STA);
            uiThread.IsBackground = true;
            uiThread.Start();

            // Wait until UI thread created the formsPlot (simple spin; you can replace with more robust sync)
            while(formsPlot == null || !formsPlot.IsHandleCreated)
                Thread.Sleep(10);
        }

        // Call this whenever you want to refresh the plot (e.g., after FitnessHistory.Add(...))
        public void UpdatePlot() {
            // guard if UI hasn't been created yet
            if(formsPlot == null || formsPlot.IsDisposed)
                return;

            // Copy the fitness history (thread-safe copy) BEFORE invoking UI thread
            List<double> fitnessCopy;
            lock(population.FitnessHistory) // lock if you also modify it elsewhere; otherwise optional
            {
                fitnessCopy = population.FitnessHistory;
            }

            formsPlot?.Invoke(() => {
                int count = fitnessCopy.Count;
                if(count == 0) {
                    return; // nothing to plot

                } else {
                    // prepare x and y data for raw fitness
                    double[] xs = Enumerable.Range(0, count).Select(i => (double)i).ToArray();
                    double[] ys = fitnessCopy.ToArray();

                    // compute smoothed series
                    var smoothed = MovingAverage(fitnessCopy, smoothingWindow);
                    double[] ysSmooth = smoothed.ToArray();

                    // average fitness history
                    var average = population.AverageFitnessHistory;
                    double[] xsAvg = Enumerable.Range(0, average.Count).Select(i => (double)i).ToArray();
                    double[] ysAvg = average.ToArray();

                    // update raw scatter
                    formsPlot.Plot.Clear();
                    scatterRaw = formsPlot.Plot.Add.Scatter(xs, ys);
                    scatterAverage = formsPlot.Plot.Add.Scatter(xsAvg, ysAvg);

                    // update smoothed scatter (if lengths differ, pad xs accordingly)
                    if(ysSmooth.Length != xs.Length) {
                        var xsSmooth = Enumerable.Range(0, ysSmooth.Length).Select(i => (double)i).ToArray();
                        scatterSmoothed = formsPlot.Plot.Add.Scatter(xsSmooth, ysSmooth);
                        formsPlot.Plot.Axes.AutoScale();
                        formsPlot.Refresh();
                    } else {
                        scatterSmoothed = formsPlot.Plot.Add.Scatter(xs, ysSmooth);
                        formsPlot.Plot.Axes.AutoScale();
                        formsPlot.Refresh();
                    }

                    scatterRaw.LegendText = "Raw Fitness";
                    scatterRaw.LineWidth = 5;
                    scatterRaw.Color = ScottColor.FromHex(AppColors.RosePink.Hex);
                    scatterRaw.MarkerSize = 2;

                    scatterAverage.LegendText = "Average Fitness";
                    scatterAverage.LineWidth = 5;
                    scatterAverage.Color = ScottColor.FromHex(AppColors.AquaGreen.Hex);
                    scatterAverage.MarkerSize = 2;
                    
                    scatterSmoothed.LegendText = "Smoothed Fitness";
                    scatterSmoothed.LineWidth = 5;
                    scatterSmoothed.Color = ScottColor.FromHex(AppColors.GoldenYellow.Hex);
                    scatterSmoothed.MarkerSize = 2;

                    CallOut("Best Fitness Ever", fitnessCopy);

                    formsPlot.Plot.ShowLegend();
                    formsPlot.Plot.Axes.AutoScale();
                    formsPlot.Refresh();
                }
            });
        }

        // Efficient sliding-window moving average (returns the same length as input)
        private static List<double> MovingAverage(List<double> data, int window) {
            var result = new List<double>(data.Count);
            if(data == null || data.Count == 0)
                return result;
            if(window <= 1) {
                result.AddRange(data);
                return result;
            }

            double sum = 0.0;
            var q = new Queue<double>();

            foreach(var v in data) {
                q.Enqueue(v);
                sum += v;

                if(q.Count > window)
                    sum -= q.Dequeue();

                result.Add(sum / q.Count);
            }

            return result;
        }

        protected virtual void Dispose(bool disposing) {
            if(!disposedValue) {
                if(disposing) {
                    // managed cleanup
                }

                try {
                    if(formsPlot != null && !formsPlot.IsDisposed) {
                        formsPlot.Invoke((Action)(() => {
                            var f = formsPlot.FindForm();
                            if(f != null && !f.IsDisposed)
                                f.Close();
                        }));
                    }
                } catch { 
                
                }

                disposedValue = true;
            }
        }

        public void Dispose() {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }

        public void CallOut(string calloutText, List<double> history) {
            if(history == null || history.Count == 0)
                return;

            // find best point
            double maxPoint = history.Max();
            int pointIndex = history.IndexOf(maxPoint);
            double bestX = pointIndex;
            double bestY = maxPoint;

            // get axis limits (after plotting, these are the current data limits)
            var limits = formsPlot.Plot.Axes.GetDataLimits(); // returns (xMin,xMax,yMin,yMax)
            double xMin = limits.Left;
            double xMax = limits.Right;
            double yMin = limits.Top;
            double yMax = limits.Bottom;

            double xRange = Math.Max(1e-6, xMax - xMin);
            double yRange = Math.Max(1e-6, yMax - yMin);

            // small offsets so the text doesn't sit directly on top of the marker
            double xOffset = xRange * 0.02; // 2% of x-range
            double yOffset = yRange * 0.04; // 4% of y-range

            // Add a visible marker at the best point (single-point scatter)
            var bestPoint = formsPlot.Plot.Add.Scatter(bestX, bestY);
            bestPoint.MarkerColor = ScottColor.FromHex("#EFC958");
            bestPoint.MarkerSize = 10;
            bestPoint.MarkerShape = MarkerShape.FilledCircle;
            bestPoint.LineWidth = 0; // no line, just a point   

            // Add text label slightly offset (to the top-right of the marker)
            string label = $"{calloutText}\nGen {pointIndex}\nVal {bestY:F2}";
            formsPlot.Plot.Add.Text(label, bestX + xOffset, bestY + yOffset);

           
            // refresh plot
            formsPlot.Plot.Axes.AutoScale(); // optional depending on whether you want autoscale
            formsPlot.Refresh();
        }

    }
}
