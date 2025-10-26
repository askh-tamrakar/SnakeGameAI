using Raylib_cs;
using ScottPlot;
using ScottPlot.Plottables;
using ScottPlot.WinForms;
using System.Numerics;
using static SnakeGameAI.Program;
using static SnakeGameAI.Snake_Game;
using Color = Raylib_cs.Color;
using Rectangle = Raylib_cs.Rectangle;
using Font = Raylib_cs.Font;
using ScottColor = ScottPlot.Color;

namespace SnakeGameAI {
    public class Game_UI {
        // ⚡ OPTIMIZATION: const for compile-time constants
        private const int offset = 25;
        private const int lineThickness = 4;

        // ⚡ OPTIMIZATION: readonly for runtime constants
        private static readonly int debugLogWidth = (int)((windowDim.X - (cellCount * cellSize + 2 * offset)) - lineThickness);
        private static readonly int debugLogHeight = (int)(windowDim.Y - (2 * offset) - (2 * 10));

        private float scrollOffset = 0;
        private int selectedIndex = -1;

        public void DrawScrollableBestGenomeList(List<Genome> bestGenomes) {
            const float itemHeight = 50f;
            int startPosY = offset + Program.offset;
            int startPosX = 982 + offset;
            int width = debugLogWidth - ((2 * offset) + Program.offset);
            const int height = 400;

            Vector2 position = new(startPosX, startPosY);
            Rectangle scrollRect = new Rectangle(position.X, position.Y, width, height);

            // Scroll wheel input
            if(Raylib.CheckCollisionPointRec(Raylib.GetMousePosition(), scrollRect)) {
                int wheel = (int)Raylib.GetMouseWheelMove();
                scrollOffset -= wheel * 20;
                float maxScroll = Math.Max(0, bestGenomes.Count * itemHeight - height);
                scrollOffset = Math.Clamp(scrollOffset, 0, maxScroll);
            }

            // Draw background
            Raylib.DrawRectangleRec(scrollRect, AppColors.ForestGreen.Ray);

            // Clip drawing to scrollRect
            Raylib.BeginScissorMode((int)scrollRect.X, (int)scrollRect.Y, (int)scrollRect.Width, (int)scrollRect.Height);

            for(int i = 0; i < bestGenomes.Count; i++) {
                int y = (int)(position.Y + i * itemHeight - scrollOffset);
                if(y + itemHeight < position.Y || y > position.Y + height)
                    continue;

                var g = bestGenomes[i];
                int score = g.IsSnakeDead ? g.Game.cachedScore : g.Score;
                string snake = g.IsSnakeDead ? "Snake is Dead" : " Snake is Alive";

                // ⚡ OPTIMIZATION: Cached string interpolation
                string text = $"ID: {g.GenomeID} | Score: {score}, Fitness: {g.Fitness:F2}, {snake}, Steps Survived: {g.StepsSnapshot}";

                Rectangle itemRect = new Rectangle(position.X, y, width, itemHeight);
                Vector2 mousePos = Raylib.GetMousePosition();
                bool isHovered = Raylib.CheckCollisionPointRec(mousePos, itemRect);
                bool isSelected = selectedIndex == i;

                Color bgColor = isSelected ? AppColors.YellowHighlight.Ray :
                    isHovered ? AppColors.JungleGreen.Ray : AppColors.ForestGreen.Ray;
                Color textColor = isSelected ? AppColors.ForestGreen.Ray :
                    isHovered ? AppColors.YellowHighlight.Ray : AppColors.YellowPastel.Ray;

                Raylib.DrawRectangleRec(itemRect, bgColor);

                // Mouse click selection
                if(isHovered && Raylib.IsMouseButtonPressed(MouseButton.Left)) {
                    selectedIndex = i;
                    population.FinalGenomeFromList = g;
                }

                if(Raylib.IsKeyPressed(KeyboardKey.Delete)) {
                    population.BestEverGenomeList.Remove(g);
                }

                Raylib.DrawTextEx(font, text, new Vector2(position.X + 10, y + 12.5f), 28, 1, textColor);
            }

            Raylib.EndScissorMode();
        }
    }

    public class LiveFitnessPlot : IDisposable {
        public FormsPlot formsPlot = null!;
        private Scatter scatterRaw = null!;
        private Scatter scatterAverage = null!;
        private Scatter scatterSmoothed = null!;
        private readonly Thread uiThread;
        private readonly Population population;
        private bool disposedValue;

        public LiveFitnessPlot(Population population) {
            this.population = population ?? throw new ArgumentNullException(nameof(population));

            uiThread = new Thread(() => {
                var form = new Form {
                    Text = "Live Fitness Plot",
                    Width = 900,
                    Height = 650
                };

                formsPlot = new FormsPlot { Dock = DockStyle.Fill };
                form.Controls.Add(formsPlot);

                // Initial empty data
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

                // Change figure colors
                formsPlot.Plot.FigureBackground.Color = ScottColor.FromHex(AppColors.MidnightNavy.Hex);
                formsPlot.Plot.DataBackground.Color = ScottColor.FromHex(AppColors.DeepSeaBlue.Hex);

                // Change axis and grid colors
                formsPlot.Plot.Axes.Color(ScottColor.FromHex(AppColors.SageGreen.Hex));
                formsPlot.Plot.Grid.MajorLineColor = ScottColor.FromHex(AppColors.SoftBlush.Hex);

                // Change legend colors
                formsPlot.Plot.Legend.BackgroundColor = ScottColor.FromHex(AppColors.MidnightNavy.Hex);
                formsPlot.Plot.Legend.FontColor = ScottColor.FromHex(AppColors.SageGreen.Hex);
                formsPlot.Plot.Legend.OutlineColor = ScottColor.FromHex(AppColors.SageGreen.Hex);

                formsPlot.Refresh();
                Application.Run(form);
            });

            uiThread.SetApartmentState(ApartmentState.STA);
            uiThread.IsBackground = true;
            uiThread.Start();

            // Wait until UI thread created the formsPlot
            while(formsPlot == null || !formsPlot.IsHandleCreated)
                Thread.Sleep(10);
        }

        public void UpdatePlot() {
            if(formsPlot == null || formsPlot.IsDisposed)
                return;

            // Copy fitness history
            List<double> fitnessCopy;
            lock(population.FitnessHistory) {
                fitnessCopy = population.FitnessHistory;
            }

            formsPlot?.Invoke(() => {
                int count = fitnessCopy.Count;
                if(count == 0) {
                    return;
                } else {
                    // Prepare x and y data
                    double[] xs = Enumerable.Range(0, count).Select(i => (double)i).ToArray();
                    double[] ys = fitnessCopy.ToArray();

                    // Compute smoothed series
                    var smoothed = population.SmoothedFitnessHistory;
                    double[] ysSmooth = smoothed.ToArray();

                    // Average fitness history
                    var average = population.AverageFitnessHistory;
                    double[] xsAvg = Enumerable.Range(0, average.Count).Select(i => (double)i).ToArray();
                    double[] ysAvg = average.ToArray();

                    // Update scatters
                    formsPlot.Plot.Clear();
                    scatterRaw = formsPlot.Plot.Add.Scatter(xs, ys);
                    scatterAverage = formsPlot.Plot.Add.Scatter(xsAvg, ysAvg);

                    if(ysSmooth.Length != xs.Length) {
                        var xsSmooth = Enumerable.Range(0, ysSmooth.Length).Select(i => (double)i).ToArray();
                        scatterSmoothed = formsPlot.Plot.Add.Scatter(xsSmooth, ysSmooth);
                    } else {
                        scatterSmoothed = formsPlot.Plot.Add.Scatter(xs, ysSmooth);
                    }

                    formsPlot.Plot.Axes.AutoScale();
                    formsPlot.Refresh();

                    var legend = formsPlot.Plot;
                    legend.Legend.Alignment = Alignment.UpperLeft;

                    scatterRaw.LegendText = "Raw Fitness";
                    scatterRaw.LineWidth = 4;
                    scatterRaw.Color = ScottColor.FromHex(AppColors.WatermelonRed.Hex);
                    scatterRaw.MarkerSize = 2;

                    scatterAverage.LegendText = "Average Fitness";
                    scatterAverage.LineWidth = 4;
                    scatterAverage.Color = ScottColor.FromHex(AppColors.AquaGreen.Hex);
                    scatterAverage.MarkerSize = 2;

                    scatterSmoothed.LegendText = "Smoothed Fitness";
                    scatterSmoothed.LineWidth = 4;
                    scatterSmoothed.Color = ScottColor.FromHex(AppColors.GoldenYellow.Hex);
                    scatterSmoothed.MarkerSize = 2;

                    CallOut("Best Fitness Ever", fitnessCopy);
                    formsPlot.Plot.ShowLegend();
                    formsPlot.Plot.Axes.AutoScale();
                    formsPlot.Refresh();
                }
            });
        }

        // ⚡ OPTIMIZATION: Efficient sliding-window moving average
        

        protected virtual void Dispose(bool disposing) {
            if(!disposedValue) {
                if(disposing) {
                    try {
                        if(formsPlot != null && !formsPlot.IsDisposed) {
                            formsPlot.Invoke((Action)(() => {
                                var f = formsPlot.FindForm();
                                if(f != null && !f.IsDisposed)
                                    f.Close();
                            }));
                        }
                    } catch { }
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

            double maxPoint = history.Max();
            int pointIndex = history.IndexOf(maxPoint);
            double bestX = pointIndex;
            double bestY = maxPoint;

            var limits = formsPlot.Plot.Axes.GetDataLimits();
            double xMin = limits.Left;
            double xMax = limits.Right;
            double yMin = limits.Top;
            double yMax = limits.Bottom;
            double xRange = Math.Max(1e-6, xMax - xMin);
            double yRange = Math.Max(1e-6, yMax - yMin);

            double xOffset = xRange * 0.02;
            double yOffset = yRange * 0.04;

            var bestPoint = formsPlot.Plot.Add.Scatter(bestX, bestY);
            bestPoint.MarkerColor = ScottColor.FromHex(AppColors.SoftBlush.Hex);
            bestPoint.MarkerSize = 10;
            bestPoint.MarkerShape = MarkerShape.FilledCircle;
            bestPoint.LineWidth = 0;

            string label = $"{calloutText}\nGen {pointIndex}\nVal {bestY:F2}";
            var infoText = formsPlot.Plot.Add.Text(label, bestX + xOffset, bestY + yOffset);
            infoText.LabelBold = true;
            infoText.LabelFontSize = 16;
            infoText.LabelFontColor = ScottColor.FromHex(AppColors.PeachCream.Hex);

            formsPlot.Plot.Axes.AutoScale();
            formsPlot.Refresh();
        }
    }
}
