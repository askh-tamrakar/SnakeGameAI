using MathNet.Numerics.LinearAlgebra;
using Raylib_cs;
using System.Numerics;
using static SnakeGameAI.Snake_Game;
using Font = Raylib_cs.Font;
using RayColor = Raylib_cs.Color;
using Rectangle = Raylib_cs.Rectangle;
using Image = Raylib_cs.Image;

namespace SnakeGameAI {

    internal class Program {
        private static int populationSize = 100;
        private static int inputSize = 20;
        private static int hiddenSize1 = 43;
        private static int hiddenSize2 = 24;
        private static int hiddenSize3 = 34;
        private static int outputSize = 3;

        public static int[] layerSizes = [
            inputSize, hiddenSize1, hiddenSize2, outputSize
        ];

        public static Population population = new(populationSize, layerSizes);
        public static LiveFitnessPlot liveFitnessPlot = new(population, smoothingWindow: 5);

        public static AI_Debugger ai_Debugger = new();
        private static Game_UI game_UI = new();
        static Game game = new();
        static Random rand = new();

        public static Font font;
        public static RayColor green => new RayColor(173, 204, 96, 255);
        public static RayColor darkGreen => new RayColor(43, 51, 24, 255);
        public static RayColor lightGreen => new RayColor(43, 75, 24, 255);

        public class DualColor {
            public string Hex { get; }
            public RayColor Ray { get; }

            public DualColor(string hex) {
                Hex = hex;
                var color = ColorTranslator.FromHtml(hex);
                Ray = new RayColor(color.R, color.G, color.B, color.A);
            }
        }

        public static class AppColors {
            // Old base colors
            public static readonly DualColor ForestGreen = new("#2F5249");
            public static readonly DualColor JungleGreen = new("#437057");
            public static readonly DualColor YellowPastel = new("#97B067");
            public static readonly DualColor YellowHighlight = new("#E3DE61");

            // Palette One
            public static readonly DualColor SageGreen = new("#99B898");
            public static readonly DualColor PeachCream = new("#FECEA8");
            public static readonly DualColor CoralPink = new("#FF847C");
            public static readonly DualColor WatermelonRed = new("#E84A5F");
            public static readonly DualColor MidnightNavy = new("#2A363B");

            // Palette Two
            public static readonly DualColor RosePink = new("#F18C8E");
            public static readonly DualColor LightPeach = new("#F0B7A4");
            public static readonly DualColor SoftBlush = new("#F1D1B5");
            public static readonly DualColor OceanTeal = new("#568EA6");
            public static readonly DualColor DeepSeaBlue = new("#305F72");

            // Palette Three
            public static readonly DualColor BrightRed = new("#EF3D59");
            public static readonly DualColor Tangerine = new("#E17A47");
            public static readonly DualColor GoldenYellow = new("#EFC958");
            public static readonly DualColor AquaGreen = new("#4AB19D");
            public static readonly DualColor StormyBlue = new("#344E5C");
        }

        public static int cellSize => 8;
        public static int cellCount => 116;
        public static int offset => 50;

        static int[] basicLatin = Enumerable.Range(0, 2048).ToArray();
        static int[] extras = {
            0x03A9,  // Ω
            0x00B1,  // ±
            0x2190,  // ←
            0x2191,  // ↑
            0x2192,  // →
            0x2193,  // ↓
            0x2194   // ↔
        };

        static int[] codepoints = basicLatin.Concat(extras).ToArray();

        static bool isVisualOn = false;
        static bool isTraining = true;
        static bool isAiPlaying = true;
        static bool isBestAiTraining = false;
        public static bool isGraphShowing = false;

        static double gameSpeed = 0.0005;
        static double lastUpdateTime = 0;

        public static Vector2 windowDim => new(1916, 1040);

        public static bool IsElementInDeque(Vector2 element, List<Vector2> deque) {
            return deque.Contains(element);
        }

        public static bool IsEventTriggered(double interval) {
            double currentTime = Raylib.GetTime();
            if(currentTime - lastUpdateTime >= interval) {
                lastUpdateTime = currentTime;
                return true;
            }
            return false;
        }

        [STAThread]
        static void Main() {
            Console.WriteLine("Starting the Game...");
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
                    if(isTraining) {
                        AI_Training();
                    } else {
                        BestAI_Snake();
                    }
                } else {
                    ManualGameMode();
                }
                Raylib.EndDrawing();
            }
            Raylib.CloseWindow();
        }

        static void LoadResources() {
            font = Raylib.LoadFontEx("Resources/NotoSans_SemiCondensed-Bold.ttf", 120, codepoints, codepoints.Length);
            AI_Debugger.font = font;

            Image icon = Raylib.LoadImage("Graphics/icon.png");
            Image snakeHead = Raylib.LoadImage("Graphics/snakeHead.png");
            Image apple = Raylib.LoadImage("Graphics/Apple.png");

            Raylib.SetWindowIcon(icon);

            Texture2D snakeHeadTexture = Raylib.LoadTextureFromImage(snakeHead);
            Texture2D appleTexture = Raylib.LoadTextureFromImage(apple);

            Raylib.UnloadImage(icon);
            Raylib.UnloadImage(snakeHead);
            Raylib.UnloadImage(apple);

            Food.texture = appleTexture;
        }

        static void DrawGameBorders() {
            int lineThickness = 5;
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

                int existingIndex = population.DeadGenomes.FindIndex(g => g.GenomeID == genome.GenomeID);
                bool exists = existingIndex != -1;

                if(!exists && genome.IsSnakeDead) {
                    population.DeadGenomes.Add(genome);
                    //DebugGenome(genome);
                }
            }

            if(population.Genomes.All(g => g.IsSnakeDead)) {
                foreach(var genome in population.Genomes) {
                    genome.CalculateFitness();
                }
                population.UpdateBestEverGenome();
                population.Evolve();
            }

            if(isGraphShowing) {  
                ai_Debugger.DrawFitnessGraph(population);
            } else {
                bestGenomeReference.HeatMap();
                game_UI.DrawScrollableBestGenomeList(population.BestEverGenomeList);
                ai_Debugger.DrawDebugPanel(bestGenomeReference, population, new Vector2(1005, 480), bestGenomeReference.GenomeID, population.BestEverGenomeList);
                Raylib.DrawTextEx(font, $"Growth {bestGenomeReference.Score}", new Vector2(offset - 5, offset + cellSize * cellCount), 60, 0, darkGreen);
            }

        }

        static void BestAI_Snake() {
            Genome currentGenome;

            if(isBestAiTraining) {
                currentGenome = population.FinalGenomeFromList;

                population.UpdateAllGenomes(isTraining);

                if(population.BestEverGenomeList.All(g => !g.Game.isRunning)) {
                    foreach(var genome in population.BestEverGenomeList) {
                        genome.CalculateFitness();
                    }

                    population.EvolveBestEverGenomes();
                }

            } else {
                currentGenome = population.FinalGenomeFromList ?? population.SmartBestGenome;

                currentGenome.HeatMap();

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
            }else {
                currentGenome.HeatMap();
                Raylib.DrawTextEx(font, $"Growth {currentGenome.Score}", new Vector2(offset - 5, offset + cellSize * cellCount), 60, 0, darkGreen);
                Raylib.DrawTextEx(font, $"Fitness {currentGenome.Fitness:F3}", new Vector2(800 - offset, offset + cellSize * cellCount), 60, 0, darkGreen);
                ai_Debugger.DrawDebugPanel(currentGenome, population, new Vector2(1005, 480), currentGenome.GenomeID, population.BestEverGenomeList);
                game_UI.DrawScrollableBestGenomeList(population.BestEverGenomeList);
            }

        }

        static void ManualGameMode() {
            if(IsEventTriggered(0.2)) {
                game.Update();
            }

            if(!game.isRunning) {
                game.isRunning = true;
            }

            UpdateManualControls();
            game.Draw();
            Raylib.DrawText($"Growth {game.score}", offset - 5, offset + cellSize * cellCount + 10, 40, darkGreen);
        }

        static void UpdateSnakeDirection(Genome currentGenome, int move) {
            Vector2 currentDir = currentGenome.Game.snake.direction;
            Vector2 left = new Vector2(-currentDir.Y, currentDir.X);    // rotate left
            Vector2 right = new Vector2(currentDir.Y, -currentDir.X);  // rotate right

            switch(move) {
            case 0: // Keep going straight
                break;
            case 1: // Turn left
                currentGenome.Game.snake.direction = left;
                break;
            case 2: // Turn right
                currentGenome.Game.snake.direction = right;
                break;
            }
        }

        static void DrawNeuralNetworkVisualization(Genome currentGenome) {
            Rectangle rect = new(offset - 5, offset - 5, windowDim.X - (offset + 30) + 10, cellSize * cellCount + 10);
            Raylib.DrawRectangle((int)rect.X + 5, (int)rect.Y + 5, (int)rect.Width - 5, (int)rect.Height - 5, RayColor.Black);
            Raylib.DrawRectangleLinesEx(rect, 5, RayColor.Red);

            ai_Debugger.DrawSlider();
            ai_Debugger.DrawNeuralNetwork(currentGenome, new Vector2((int)rect.X + 5, (int)rect.Y + 5), (int)rect.Width - 5, (int)rect.Height - 5);
        }

        static void UpdateManualControls() {
            if(Raylib.IsKeyPressed(KeyboardKey.Up) && game.snake.direction.Y != 1) {
                game.snake.direction = new Vector2(0, -1);
            }
            if(Raylib.IsKeyPressed(KeyboardKey.Down) && game.snake.direction.Y != -1) {
                game.snake.direction = new Vector2(0, 1);
            }
            if(Raylib.IsKeyPressed(KeyboardKey.Left) && game.snake.direction.X != 1) {
                game.snake.direction = new Vector2(-1, 0);
            }
            if(Raylib.IsKeyPressed(KeyboardKey.Right) && game.snake.direction.X != -1) {
                game.snake.direction = new Vector2(1, 0);
            }
        }

        static void ControlOptions() {
            // Press F1 to toggle LogScale
            if(Raylib.IsKeyPressed(KeyboardKey.F1)) {
                ai_Debugger.isLogScale = !ai_Debugger.isLogScale;
                Console.WriteLine("Switched to " + (ai_Debugger.isLogScale ? "Logarithmic" : "Linear") + " scale");
            }

            // Press F2 to toggle AI Training Mode
            if(Raylib.IsKeyPressed(KeyboardKey.F2)) {
                isTraining = !isTraining;
                Console.WriteLine(isTraining ? "Training Mode" : "Best Genome Mode");
            }

            // Press F3 to toggle Neural Network Visual 
            if(Raylib.IsKeyPressed(KeyboardKey.F3)) {
                if(!isTraining) {
                    isVisualOn = !isVisualOn;
                    Console.WriteLine(isVisualOn ? "Neural Visual ON" : "Neural Visual OFF");
                } else { 
                    isGraphShowing = !isGraphShowing;
                    Console.WriteLine(isGraphShowing ? "Graph is Showing" : "Graph is Hidded");
                }
            }

            // Press T to change Best AI Train Mode
            if(Raylib.IsKeyPressed(KeyboardKey.T)) {
                isBestAiTraining = !isBestAiTraining;
                Console.WriteLine(isBestAiTraining ? "Best AI Training Mode" : "Snake from Best Ever Genome List is Playing");
            }

            // Press S to Save Neural Network Size
            if(Raylib.IsKeyPressed(KeyboardKey.S)) {
                ai_Debugger.SaveSettings();
            }

            // Press L to Load Neural Network Size
            if(Raylib.IsKeyPressed(KeyboardKey.L)) {
                ai_Debugger.LoadSettings();
            }

            // Press TAB to Switch Game Mode
            if(Raylib.IsKeyPressed(KeyboardKey.Tab)) {
                isAiPlaying = !isAiPlaying;
                Console.WriteLine(isAiPlaying ? "AI Game Mode" : "Manual Game Mode");
            }

            // Press F5 to Save BestGenome List
            if(Raylib.IsKeyDown(KeyboardKey.F5)) {
                Genome_Persistence.SaveGenerationOrBestEverList(population, "Save States/Best Genomes.json");
            }

            // Press F9 to Load BestGenome List
            if(Raylib.IsKeyDown(KeyboardKey.F9)) {
                (population.Genomes, population.BestEverGenomeList) = Genome_Persistence.LoadGenerationOrBestEverList(population, "Save States/Best Genomes.json");
            }
        }

        public static void DebugGenome(Genome genome) {
            int score = genome.IsSnakeDead ? genome.Game.updateScore : genome.Score;
            string snake = genome.IsSnakeDead ? "Snake is Dead" : " Snake is Alive";

            Console.WriteLine($"ID: {genome.GenomeID} | " +
                              $"Sore: {genome.Game.updateScore} | " +
                              $"Fitness: {genome.Fitness} | " +
                              $"{snake} | " +
                              $"Score: {score}");

        }
    }
}
