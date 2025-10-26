using MathNet.Numerics.LinearAlgebra;
using Raylib_cs;
using System.Numerics;
using static SnakeGameAI.Snake_Game;
using Font = Raylib_cs.Font;
using RayColor = Raylib_cs.Color;
using System.Drawing;
using Rectangle = Raylib_cs.Rectangle;
using Image = Raylib_cs.Image;

namespace SnakeGameAI {
    internal class Program {
        // ⚡ OPTIMIZED HYPERPARAMETERS
        private static readonly int populationSize = 100;
        private static readonly int inputSize = 37;
        private static readonly int hiddenSize1 = 32;      // Reduced from 43
        private static readonly int hiddenSize2 = 16;      // Reduced from 24
        private static readonly int outputSize = 3;
        public static readonly int[] layerSizes = [
            inputSize, hiddenSize1, hiddenSize2, outputSize
        ];

        public static Population population = new(populationSize, layerSizes);
        public static LiveFitnessPlot liveFitnessPlot = new(population);
        public static AI_Debugger ai_Debugger = new();
        private static Game_UI game_UI = new();
        static Game game = new();

        public static Font font;

        // ⚡ OPTIMIZATION: Made readonly for better performance
        public static readonly RayColor green = new(173, 204, 96, 255);
        public static readonly RayColor darkGreen = new(43, 51, 24, 255);
        public static readonly RayColor lightGreen = new(43, 75, 24, 255);

        // ⚡ OPTIMIZATION: Changed from class to readonly struct for better memory layout
        public readonly struct DualColor {
            public string Hex { get; }
            public RayColor Ray { get; }

            public DualColor(string hex) {
                Hex = hex;
                var color = ColorTranslator.FromHtml(hex);
                Ray = new RayColor(color.R, color.G, color.B, color.A);
            }
        }

        public static class AppColors {
            public static readonly DualColor ForestGreen = new("#2F5249");
            public static readonly DualColor JungleGreen = new("#437057");
            public static readonly DualColor YellowPastel = new("#97B067");
            public static readonly DualColor YellowHighlight = new("#E3DE61");
            public static readonly DualColor SageGreen = new("#99B898");
            public static readonly DualColor PeachCream = new("#FECEA8");
            public static readonly DualColor CoralPink = new("#FF847C");
            public static readonly DualColor WatermelonRed = new("#E84A5F");
            public static readonly DualColor MidnightNavy = new("#2A363B");
            public static readonly DualColor RosePink = new("#F18C8E");
            public static readonly DualColor LightPeach = new("#F0B7A4");
            public static readonly DualColor SoftBlush = new("#F1D1B5");
            public static readonly DualColor OceanTeal = new("#568EA6");
            public static readonly DualColor DeepSeaBlue = new("#305F72");
            public static readonly DualColor BrightRed = new("#EF3D59");
            public static readonly DualColor Tangerine = new("#E17A47");
            public static readonly DualColor GoldenYellow = new("#EFC958");
            public static readonly DualColor AquaGreen = new("#4AB19D");
            public static readonly DualColor StormyBlue = new("#344E5C");
        }

        public static int cellSize = 16;
        public static int cellCount = 58;
        public static int offset = 50;
        public static bool isGhostMode = false;
        public static bool isGraphShowing = false;
        static bool isBestAiTraining = false;
        public static Vector2 windowDim = new(1916, 1040);

        // ⚡ OPTIMIZATION: Made readonly and static
        static readonly int[] basicLatin = Enumerable.Range(0, 2048).ToArray();
        static readonly int[] extras = { 0x03A9, 0x00B1, 0x2190, 0x2191, 0x2192, 0x2193, 0x2194 };
        static readonly int[] codepoints = basicLatin.Concat(extras).ToArray();

        static bool isVisualOn = false;
        static bool isTraining = true;
        static bool isAiPlaying = true;

        static double gameSpeed = 0.0005;
        static double lastUpdateTime = 0;

        // ⚡ OPTIMIZATION: Inline method
        public static bool IsElementInDeque(Vector2 element, List<Vector2> deque) => deque.Contains(element);

        public static bool IsEventTriggered(double interval) {
            double currentTime = Raylib.GetTime();
            if(currentTime - lastUpdateTime >= interval) {
                lastUpdateTime = currentTime;
                return true;
            }
            return false;
        }

        [STAThread]
        static void Main(string[] args) {
            TrainingLogger.Initialize();

            TrainingLogger.Log("Starting the Game...");
            TrainingLogger.Log("===== SNAKE AI - NEURAL EVOLUTION =====");
            TrainingLogger.Log($"Population: {populationSize} | Architecture: {string.Join("-", layerSizes)}");
            TrainingLogger.Log($"Elite: 10% | Tournament: 7| Immigration: 5%");
            TrainingLogger.Log("================================================\n");

            Raylib.InitWindow((int)windowDim.X, (int)windowDim.Y, "AI Powered Retro Snake");
            Raylib.SetTargetFPS(60);
            ai_Debugger.LoadSettings();
            LoadResources();

            while(!Raylib.WindowShouldClose()) {
                Raylib.BeginDrawing();
                Raylib.ClearBackground(green);
                DrawGameBorders();
                DrawGameTitle();
                ControlOptions();

                ai_Debugger.graphZoom += Raylib.GetMouseWheelMove() * 0.1f;
                ai_Debugger.graphZoom = Math.Clamp(ai_Debugger.graphZoom, 0.2f, 5f);

                if(isAiPlaying) {
                    if(isTraining)
                        AI_Training();
                    else
                        BestAI_Snake();
                } else {
                    ManualGameMode();
                }

                Raylib.EndDrawing();
            }

            TrainingLogger.LogTrainingSummary();
            Raylib.CloseWindow();
        }

        static void LoadResources() {
            font = Raylib.LoadFontEx("Resources/NotoSans_SemiCondensed-Bold.ttf", 120, codepoints, codepoints.Length);
            AI_Debugger.font = font;
            Image icon = Raylib.LoadImage("Graphics/icon.png");
            Image snakeHead = Raylib.LoadImage("Graphics/snakeHead.png");
            Image apple = Raylib.LoadImage("Graphics/Apple.png");
            Raylib.SetWindowIcon(icon);
            Texture2D appleTexture = Raylib.LoadTextureFromImage(apple);
            Raylib.UnloadImage(icon);
            Raylib.UnloadImage(snakeHead);
            Raylib.UnloadImage(apple);
            Food.texture = appleTexture;
        }

        static void DrawGameBorders() {
            const int lineThickness = 5;
            int startPosX = offset - lineThickness;
            int startPosY = offset - lineThickness;
            int width = (int)(windowDim.X - (2 * offset)) + 2 * lineThickness;
            int height = (cellSize * cellCount) + 10;
            Raylib.DrawRectangleLinesEx(new Rectangle(startPosX, startPosY, width, height), lineThickness, darkGreen);
            Raylib.DrawLineEx(new Vector2(980, 50), new Vector2(980, 980), lineThickness, darkGreen);
        }

        static void DrawGameTitle() {
            Raylib.DrawText("Retro Snake - ", offset - 5, 8, 40, darkGreen);
            Raylib.DrawText("Best Genome Showcase", 375, 15, 32, lightGreen);
        }

        static void AI_Training() {
            Genome bestGenomeReference = population.SmartBestGenome;
            population.UpdateAllGenomes();

            for(int i = 0; i < population.Genomes.Count; i++) {
                var genome = population.Genomes[i];
                if(!genome.IsSnakeDead)
                    continue;

                int existingIndex = population.DeadGenomes.FindIndex(g => g.GenomeID == genome.GenomeID);
                if(existingIndex == -1) {
                    population.DeadGenomes.Add(genome);
                }
            }

            if(population.Genomes.All(g => g.IsSnakeDead)) {
                foreach(var genome in population.Genomes) {
                    genome.CalculateFitness();

                    TrainingLogger.LogGenomeDeath(
                        genome,
                        population.Generation,
                        addedToBest: false
                    );
                }
                population.UpdateBestEverGenome();
                TrainingLogger.LogGeneration(population);
                population.Evolve();
            }

            if(isGraphShowing) {
                ai_Debugger.DrawFitnessGraph(population);
            } else {
                game_UI.DrawScrollableBestGenomeList(population.BestEverGenomeList);
                ai_Debugger.DrawDebugPanel(bestGenomeReference, population, new Vector2(1005, 480),
                    bestGenomeReference.GenomeID, population.BestEverGenomeList);
                Raylib.DrawTextEx(font, $"Growth {bestGenomeReference.Score}",
                    new Vector2(offset - 5, offset + cellSize * cellCount), 60, 0, darkGreen);
            }
        }

        static void BestAI_Snake() {
            Genome currentGenome;

            if(isBestAiTraining) {
                currentGenome = population.FinalGenomeFromList;
                population.UpdateAllGenomes(isTraining: false);

                if(population.BestEverGenomeList.All(g => !g.Game.isRunning)) {
                    foreach(var genome in population.BestEverGenomeList)
                        genome.CalculateFitness();
                    population.EvolveBestEverGenomes();
                }
            } else {
                currentGenome = population.FinalGenomeFromList ?? population.SmartBestGenome;

                if(!currentGenome.Game.isRunning) {
                    currentGenome.Game.Reset();
                    currentGenome.Game.isRunning = true;
                }

                if(IsEventTriggered(gameSpeed)) {
                    double[] inputs = currentGenome.Game.GetInputs();
                    Matrix<double> inputMatrix = Matrix<double>.Build.Dense(1, inputs.Length, (i, j) => inputs[j]);
                    var (_, move) = currentGenome.NeuralNetwork.Predict(inputMatrix);
                    UpdateSnakeDirection(currentGenome, move);
                    currentGenome.Game.Update();
                }

                currentGenome.Game.Draw();
            }

            if(isVisualOn) {
                DrawNeuralNetworkVisualization(currentGenome);
            } else {
                Raylib.DrawTextEx(font, $"Growth {currentGenome.Score}",
                    new Vector2(offset - 5, offset + cellSize * cellCount), 60, 0, darkGreen);
                Raylib.DrawTextEx(font, $"Fitness {currentGenome.Fitness:F3}",
                    new Vector2(800 - offset, offset + cellSize * cellCount), 60, 0, darkGreen);
                ai_Debugger.DrawDebugPanel(currentGenome, population, new Vector2(1005, 480),
                    currentGenome.GenomeID, population.BestEverGenomeList);
                game_UI.DrawScrollableBestGenomeList(population.BestEverGenomeList);
            }
        }

        static void ManualGameMode() {
            if(IsEventTriggered(0.2))
                game.Update();
            if(!game.isRunning)
                game.isRunning = true;

            UpdateManualControls();
            game.Draw();
            Raylib.DrawText($"Growth {game.score}", offset - 5, offset + cellSize * cellCount + 10, 40, darkGreen);
        }

        static void UpdateSnakeDirection(Genome currentGenome, int move) {
            Vector2 currentDir = currentGenome.Game.snake.direction;
            Vector2 left = new(-currentDir.Y, currentDir.X);
            Vector2 right = new(currentDir.Y, -currentDir.X);

            switch(move) {
            case 0:
                break;
            case 1:
                currentGenome.Game.snake.direction = left;
                break;
            case 2:
                currentGenome.Game.snake.direction = right;
                break;
            }
        }

        static void DrawNeuralNetworkVisualization(Genome currentGenome) {
            Rectangle rect = new(offset - 5, offset - 5, windowDim.X - (offset + 30) + 10, cellSize * cellCount + 10);
            Raylib.DrawRectangle((int)rect.X + 5, (int)rect.Y + 5, (int)rect.Width - 5, (int)rect.Height - 5, RayColor.Black);
            Raylib.DrawRectangleLinesEx(rect, 5, AppColors.BrightRed.Ray);
            ai_Debugger.DrawNeuralNetwork(currentGenome, new Vector2((int)rect.X + 5, (int)rect.Y + 5),
                (int)rect.Width - 5, (int)rect.Height - 5);
        }

        static void UpdateManualControls() {
            if(Raylib.IsKeyPressed(KeyboardKey.Up) && game.snake.direction.Y != 1)
                game.snake.direction = new Vector2(0, -1);
            if(Raylib.IsKeyPressed(KeyboardKey.Down) && game.snake.direction.Y != -1)
                game.snake.direction = new Vector2(0, 1);
            if(Raylib.IsKeyPressed(KeyboardKey.Left) && game.snake.direction.X != 1)
                game.snake.direction = new Vector2(-1, 0);
            if(Raylib.IsKeyPressed(KeyboardKey.Right) && game.snake.direction.X != -1)
                game.snake.direction = new Vector2(1, 0);
        }

        static void ControlOptions() {
            if(Raylib.IsKeyPressed(KeyboardKey.F1)) {
                ai_Debugger.isLogScale = !ai_Debugger.isLogScale;
                Console.WriteLine("Switched to " + (ai_Debugger.isLogScale ? "Logarithmic" : "Linear") + " scale");
            }

            if(Raylib.IsKeyPressed(KeyboardKey.F2)) {
                isTraining = !isTraining;
                Console.WriteLine(isTraining ? "Training Mode" : "Best Genome Mode");
            }

            if(Raylib.IsKeyPressed(KeyboardKey.F3)) {
                if(!isTraining) {
                    isVisualOn = !isVisualOn;
                    Console.WriteLine(isVisualOn ? "Neural Visual ON" : "Neural Visual OFF");
                } else {
                    isGraphShowing = !isGraphShowing;
                    Console.WriteLine(isGraphShowing ? "Graph is Showing" : "Graph is Hidden");
                }
            }

            if(Raylib.IsKeyPressed(KeyboardKey.T)) {
                isBestAiTraining = !isBestAiTraining;
                Console.WriteLine(isBestAiTraining ? "Best AI Training Mode" : "Snake from Best Ever Genome List is Playing");
            }

            if(Raylib.IsKeyPressed(KeyboardKey.Backspace)) {
                gameSpeed = Math.Max(0.0001, gameSpeed - 0.0001);
                Console.WriteLine($"Game Speed: {gameSpeed:F4} seconds per move");
            }

            if(Raylib.IsKeyPressed(KeyboardKey.Minus)) {
                gameSpeed += 0.0001;
                Console.WriteLine($"Game Speed: {gameSpeed:F4} seconds per move");
            }

            if(Raylib.IsKeyPressed(KeyboardKey.Equal)) {
                gameSpeed -= 0.0001;
                Console.WriteLine($"Game Speed: {gameSpeed:F4} seconds per move");
            }

            if(Raylib.IsKeyPressed(KeyboardKey.G)) {
                isGhostMode = !isGhostMode;
                Console.WriteLine(isGhostMode ? "Entered Ghost Mode" : "Left Ghost Mode");
            }

            if(Raylib.IsKeyPressed(KeyboardKey.S))
                ai_Debugger.SaveSettings();

            if(Raylib.IsKeyPressed(KeyboardKey.L))
                ai_Debugger.LoadSettings();

            if(Raylib.IsKeyPressed(KeyboardKey.Tab)) {
                isAiPlaying = !isAiPlaying;
                Console.WriteLine(isAiPlaying ? "AI Game Mode" : "Manual Game Mode");
            }

            if(Raylib.IsKeyDown(KeyboardKey.F5)) {
                Genome_Persistence.SaveGenerationOrBestEverList(population, "Save States/Best Genomes.json");
            }

            if(Raylib.IsKeyDown(KeyboardKey.F9)) {
                (population.Genomes, population.BestEverGenomeList) =
                    Genome_Persistence.LoadGenerationOrBestEverList(population, "Save States/Best Genomes.json");
            }
        }

        public static void DebugGenome(Genome genome) {
            int score = genome.IsSnakeDead ? genome.Game.cachedScore : genome.Score;
            string snake = genome.IsSnakeDead ? "Snake is Dead" : "Snake is Alive";
            Console.WriteLine($"ID: {genome.GenomeID} | Score: {genome.Game.cachedScore} | Fitness: {genome.Fitness} | {snake} | Score: {score}");
        }
    }
}
