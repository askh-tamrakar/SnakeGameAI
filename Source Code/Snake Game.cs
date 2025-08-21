using MathNet.Numerics.LinearAlgebra;
using Raylib_cs;
using System.Diagnostics;
using System.Numerics;
using static SnakeGameAI.Program;
using static SnakeGameAI.Snake_Game;
using Font = Raylib_cs.Font;
using Color = Raylib_cs.Color;
using Rectangle = Raylib_cs.Rectangle;

namespace SnakeGameAI {
    public class Snake_Game {
        private static readonly Color DarkGreen = new Color(43, 51, 24, 255);

        public static Vector2 GenerateRandomCellFood() {
            int x = Raylib.GetRandomValue(0, cellCount - 1);
            int y = Raylib.GetRandomValue(0, cellCount - 1);
            return new Vector2(x, y);
        }
        public static Vector2 GenerateRandomCellSnake() {
            int x = Raylib.GetRandomValue(5, cellCount - 5);
            int y = Raylib.GetRandomValue(5, cellCount - 5);
            return new Vector2(x, y);
        }

        public static Vector2 GenerateRandomPos(List<Vector2> snakeBody) {
            Vector2 pos = GenerateRandomCellFood();
            while(IsElementInDeque(pos, snakeBody)) {
                pos = GenerateRandomCellFood();
            }
            return pos;
        }

        public class Genome {
            public NeuralNetwork NeuralNetwork { get; }
            public Game Game { get; }

            public int StepsSnapshot {
                get => Game.stepsSurvived;
                set => Game.stepsSurvived = value;
            }
            public int Score => Game.score;
            public int SnakeLength => Game.snake.body.Count;

            public bool HasFinalFitness { get; set; } = false;
            public bool IsSnakeDead => !Game.isRunning;

            public string GenomeID { get; set; } = string.Empty;

            public double Fitness { get; set; }

            public void AssignID(int generation, int index) {
                string generationHex = generation.ToString("X3"); // 3-digit hex
                string indexHex = index.ToString("X3");      // 3-digit hex
                GenomeID = $"{generationHex}{indexHex}";
            }

            public Genome(NeuralNetwork neuralNetwork, Game? game = null) {
                NeuralNetwork = neuralNetwork;
                Game = game ?? new Game(true);
            }

            public Genome(int[] layerSizes)
                : this(new NeuralNetwork(layerSizes), null) {
                Fitness = 0;
            }

            public Genome CloneAndMutate(double mutationRate) {
                var clone = new Genome(NeuralNetwork.Clone());
                clone.Mutate(mutationRate);
                return clone;
            }

            public void Mutate(double mutationRate) =>
                NeuralNetwork.Mutate(mutationRate);

            public Genome DeepClone() {
                var clonedNet = NeuralNetwork.Clone();
                var clonedGame = new Game(headless: true) {
                    snake = {
                        body = new List<Vector2>(Game.snake.body),
                        direction = Game.snake.direction,
                        isAddSegment = Game.snake.isAddSegment
                    },
                    score = Game.score,
                    updateScore = Game.updateScore,
                    isRunning = Game.isRunning
                };
                clonedGame.food.position = Game.food.position;

                var clone = new Genome(clonedNet, clonedGame) {
                    Fitness = this.Fitness,
                    GenomeID = this.GenomeID,
                    StepsSnapshot = this.StepsSnapshot,

                };
                return clone;
            }

            public Genome ShallowClone() {
                var clonedNet = NeuralNetwork.Clone();

                var clone = new Genome(clonedNet) {
                    GenomeID = this.GenomeID,
                    StepsSnapshot = this.StepsSnapshot
                };

                return clone;
            }

            public void CalculateFitness() {
                if(HasFinalFitness)
                    return;

                var headPosition = Game.snake.body[0];
                var applePosition = Game.food.position;
                float distanceToApple = Vector2.Distance(headPosition, applePosition);

                double fitness = 0;

                 // 🎯 BONUSES
                // Reward on the bsis of Setps
                double stepsLeft = Game.maxStepsWithoutFood - Game.stepsSinceLastFood;
                double stepsLeftNormalized = stepsLeft / (double)Game.maxStepsWithoutFood;
                double eatReward = 1.0 + (stepsLeftNormalized * 0.5);
                fitness += eatReward;

                // 🧠 Strategic movement bonus (toward apple)
                float dx = applePosition.X - headPosition.X;
                float dy = applePosition.Y - headPosition.Y;
                if((dx > 0 && Game.snake.direction.X == 1) || (dx < 0 && Game.snake.direction.X == -1))
                    fitness += 1.0;
                if((dy > 0 && Game.snake.direction.Y == 1) || (dy < 0 && Game.snake.direction.Y == -1))
                    fitness += 1.0;

                // 🍎 Base reward for eating
                fitness += Game.updateScore * 150;


                // 🚀 Boost based on distance snake had to cover to reach food
                if(Game.startDistanceToFood > 0 && Score > 0) {
                    fitness += Game.startDistanceToFood * Score * 10;
                }

                // ⏱️ Reward for surviving
                fitness += Game.stepsSurvived * 0.1;

                // 📏 Bonus for snake length
                fitness += Math.Pow(SnakeLength, 1.3);

                 // ❌ PENALTIES
                // 📉 Penalty for being far from food
                fitness -= distanceToApple * 0.05;

                // Penalty for Collision Death
                if(Game.causeOfDeath == Game.DeathCause.CollidedToBody)
                    fitness -= 50;
                if(Game.causeOfDeath == Game.DeathCause.CollidedToWall)
                    fitness -= 20;

                // 🧠 Penalty for bad or non-progressing behavior
                if(Score == 0 || IsSnakeDead) {
                    fitness *= 0.3;
                }

                Fitness = fitness;
            }

            public void Step(Game game, int move) {
                if (!game.isRunning) return;

                if(IsEventTriggered(0.002)) {
                    Matrix<double> input = Matrix<double>.Build.Dense(1, game.GetInputs().Length, (i, j) => game.GetInputs()[j]);
                    (_, move) = NeuralNetwork.Predict(input);

                    // Determine new direction relative to current one
                    Vector2 currentDir = game.snake.direction;
                    Vector2 left = new Vector2(-currentDir.Y, currentDir.X);   // rotate left
                    Vector2 right = new Vector2(currentDir.Y, -currentDir.X);  // rotate right

                    switch(move) {
                        case 0: // keep going straight
                            break;
                        case 1: // turn left
                            game.snake.direction = left;
                            break;
                        case 2: // turn right
                            game.snake.direction = right;
                            break;
                    }
                    game.Update();
                }
                game.Draw();
            }

            public void HeatMap() {
                int heatmapX = 1300;
                int heatmapY = 525;
                int cellSize = 10;

                // Layer One
                Raylib.DrawText("Layer 1 Weight Mutations", heatmapX, heatmapY - 20, 20, Color.White);
                AI_Debugger.DrawHeatmap(NeuralNetwork.CumulativeMutations[0], heatmapX, heatmapY, cellSize);

                Raylib.DrawText("Layer 1 Bias Mutations", heatmapX, heatmapY + 30 + NeuralNetwork.GetLayers()[0].Weights.RowCount * cellSize, 20, Color.White);
                AI_Debugger.DrawHeatmap(NeuralNetwork.CumulativeMutations[1], heatmapX, heatmapY + 60 + NeuralNetwork.GetLayers()[2].Weights.RowCount * cellSize, cellSize);

                //Layer Two
                int heatmapX_2 = heatmapX;
                Raylib.DrawText("Layer 1 Weight Mutations", heatmapX + 260, heatmapY - 20, 20, Color.White);
                AI_Debugger.DrawHeatmap(NeuralNetwork.CumulativeMutations[2], heatmapX + 260, heatmapY, cellSize);

                Raylib.DrawText("Layer 1 Bias Mutations", heatmapX + 260, heatmapY + 30 + NeuralNetwork.GetLayers()[0].Weights.RowCount * cellSize, 20, Color.White);
                AI_Debugger.DrawHeatmap(NeuralNetwork.CumulativeMutations[3], heatmapX + 260, heatmapY + 60 + NeuralNetwork.GetLayers()[2].Weights.RowCount * cellSize, cellSize);
            }
        }

        public class Snake {
            public List<Vector2> body;
            public Vector2 direction;
            public bool isAddSegment = false;

            public Snake() {
                Vector2 start = GenerateRandomCellSnake();
                int bodyX = (int)start.X;
                int bodyY = (int)start.Y;

                direction = new Vector2(1, 0);
                body = new List<Vector2>{
                    start,
                    new Vector2(bodyX - 1, bodyY),
                    new Vector2(bodyX - 2, bodyY)
                };
            }


            public void Draw() {
                foreach(var segment in body) {
                    Rectangle rect = new Rectangle(
                        offset + segment.X * cellSize,
                        offset + segment.Y * cellSize,
                        cellSize, cellSize
                    );
                    Raylib.DrawRectangleRounded(rect, 0.5f, 6, DarkGreen);
                }

            }

            public void Update() {
                if(!isAddSegment) {
                    body.RemoveAt(body.Count - 1);
                } else {
                    isAddSegment = false;
                }
                body.Insert(0, Vector2.Add(body[0], direction));
            }

            public void Reset() {
                Vector2 start = GenerateRandomCellSnake();
                int bodyX = (int)start.X;
                int bodyY = (int)start.Y;

                body = new List<Vector2> {
                    start,
                    new Vector2(bodyX - 1, bodyY),
                    new Vector2(bodyX - 2, bodyY)
                };

                direction = new Vector2(1, 0);
                isAddSegment = false;
            }
        }

        public class Food {
            public Vector2 position;
            public static Texture2D? texture;
            public bool isHeadless;

            public Food(List<Vector2> snakeBody, bool headless = false) {
                isHeadless = headless;
                position = GenerateRandomPos(snakeBody);
            }

            public void Draw() {
                Rectangle dest = new Rectangle(
                    offset + position.X * cellSize,
                    offset + position.Y * cellSize,
                    cellSize,
                    cellSize
                );

                Rectangle source = new Rectangle(0, 0, texture!.Value.Width, texture.Value.Height);
                Vector2 origin = new Vector2(0, 0);
                Raylib.DrawTexturePro(texture.Value, source, dest, origin, 0f, Color.White);
            }

        }

        public class Game {
            public Snake snake = new Snake();
            public Food food;

            public int score = 0;
            public int updateScore = 0;
            public int stepsSinceLastFood = 0;
            public int maxStepsWithoutFood = 200;
            public int stepsSurvived = 0;

            public float startDistanceToFood = 0;

            public bool isRunning = true;
            public bool isHeadless;

            public static Sound eatSound;
            public static Sound wallSound;

            public enum DeathCause { 
                CollidedToBody,
                CollidedToWall
            }

            public DeathCause causeOfDeath;
            public Game(bool headless = false) {
                isHeadless = headless;
                food = new Food(snake.body, isHeadless);

                if(!isHeadless) {
                    Raylib.InitAudioDevice();
                    eatSound = Raylib.LoadSound("Sound/eat.mp3");
                    wallSound = Raylib.LoadSound("Sound/wall.mp3");
                }
            }

            ~Game() {
                if(!isHeadless ) {
                    Raylib.UnloadSound(eatSound);
                    Raylib.UnloadSound(wallSound);
                    Raylib.CloseAudioDevice();
                }
            }

            public void PlaySoundSafe(Sound sound) {
                Raylib.PlaySound(sound);
            }


            public void Draw() {
                food.Draw();
                snake.Draw();
            }

            public void ClearFood() {
                food.position = new Vector2(-1000, -1000);
            }

            public void CheckCollisionWithFood() {
                if(snake.body[0].Equals(food.position)) {
                    food.position = GenerateRandomPos(snake.body);
                    startDistanceToFood = Vector2.Distance(snake.body[0], food.position);
                    snake.isAddSegment = true;
                    PlaySoundSafe(eatSound);
                    stepsSinceLastFood = 0;
                    score++;
                }
            }

            public void CheckCollisionWithEdges() {
                if(snake.body[0].X >= cellCount || snake.body[0].X < 0 ||
                    snake.body[0].Y >= cellCount || snake.body[0].Y < 0) {
                    GameOver();
                    PlaySoundSafe(wallSound);
                }

                causeOfDeath = DeathCause.CollidedToBody;
            }

            public void CheckCollisionWithTail() {
                List<Vector2> headlessBody = new List<Vector2>(snake.body);
                headlessBody.RemoveAt(0);
                if(IsElementInDeque(snake.body[0], headlessBody)) {
                    GameOver();
                    PlaySoundSafe(wallSound);
                }

                causeOfDeath = DeathCause.CollidedToWall;
            }

            private bool IsCollision(Vector2 position) {
                // Wall collision
                if(position.X < 0 || position.Y < 0 || position.X >= cellCount || position.Y >= cellCount)
                    return true;

                // Self-collision
                return snake.body.Skip(1).Any(segment => segment == position);
            }

            bool IsInsideGrid(Vector2 pos) {
                return pos.X < 0 || pos.Y < 0 || pos.X >= cellCount || pos.Y >= cellCount;
            }

            public void Update() {
                if(isRunning) {
                    stepsSurvived++;
                    stepsSinceLastFood++;
                    snake.Update();
                    isRunning = true;
                    CheckCollisionWithFood();
                    CheckCollisionWithEdges();
                    CheckCollisionWithTail();


                    if(Raylib.IsKeyPressed(KeyboardKey.K)) {
                        GameOver();
                    }

                    if(Raylib.IsKeyPressed(KeyboardKey.R)) {
                        Reset();
                    }

                    if(stepsSinceLastFood >= (maxStepsWithoutFood + (score * 8))) {
                        GameOver();
                    }
                }
            }

            public void GameOver() {
                snake.Reset();
                food.position = GenerateRandomPos(snake.body);
                isRunning = false;
                stepsSinceLastFood = 0;
                updateScore = score;
                score = 0;
            }

            public void Reset() {
                snake = new Snake();
                food = new Food(snake.body);
                stepsSurvived = 0;
                score = 0;
                stepsSinceLastFood = 0;
                isRunning = true;
            }

            public double[] GetInputs() {
                List<double> inputs = new();

                var head = snake.body[0];
                Vector2 dir = snake.direction;
                Vector2 left = new Vector2(-dir.Y, dir.X);
                Vector2 right = new Vector2(dir.Y, -dir.X);
                var foodPos = food.position;

                // Normalized direction [-1, 1]
                inputs.Add(dir.X);
                inputs.Add(dir.Y);

                // Danger checks (wall/body collisions)
                inputs.Add(IsCollision(head + dir) ? 1.0 : 0.0);
                inputs.Add(IsCollision(head + left) ? 1.0 : 0.0);
                inputs.Add(IsCollision(head + right) ? 1.0 : 0.0);

                // Food direction (relative to facing)
                Vector2 toFood = Vector2.Normalize(foodPos - head);
                inputs.Add(Vector2.Dot(dir, toFood));     // Ahead
                inputs.Add(Vector2.Dot(left, toFood));    // Left
                inputs.Add(Vector2.Dot(right, toFood));   // Right

                // -------- Advanced Snake Body Inputs --------
                Vector2[] directions = new Vector2[] {
                    new(0, -1),   // Up
                    new(1, -1),   // Up-Right
                    new(1, 0),    // Right
                    new(1, 1),    // Down-Right
                    new(0, 1),    // Down
                    new(-1, 1),   // Down-Left
                    new(-1, 0),   // Left
                    new(-1, -1),  // Up-Left
                };

                int maxVision = 10;

                foreach(var visionDir in directions) {
                    double bodyVision = 0;
                    for(int step = 1;step <= maxVision;step++) {
                        var check = head + visionDir * step;
                        if(IsCollision(check))
                            break;

                        if(snake.body.Contains(check)) {
                            bodyVision = 1.0 - (step - 1) / (double)(maxVision - 1);
                            break;
                        }
                    }
                    inputs.Add(bodyVision);
                }

                // -------- Global Food Direction (Absolute) --------
                inputs.Add(foodPos.Y < head.Y ? 1.0 : 0.0); // Food is up
                inputs.Add(foodPos.Y > head.Y ? 1.0 : 0.0); // Food is down
                inputs.Add(foodPos.X < head.X ? 1.0 : 0.0); // Food is left
                inputs.Add(foodPos.X > head.X ? 1.0 : 0.0); // Food is right

                return inputs.ToArray();
            }

        }
    }
}
