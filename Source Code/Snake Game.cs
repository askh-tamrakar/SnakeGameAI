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
            public int CachedScore => Game.cachedScore;
            public int SnakeLength => Game.snake.body.Count;
            public bool HasFinalFitness { get; set; } = false;
            public bool IsSnakeDead => !Game.isRunning;
            public string GenomeID { get; set; } = string.Empty;
            public double Fitness { get; set; }

            public void AssignID(int generation, int index) {
                string generationHex = generation.ToString("X3");
                string indexHex = index.ToString("X3");
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

            // ⚡ OPTIMIZATION: Reduced allocations in DeepClone
            public Genome DeepClone() {
                var clonedNet = NeuralNetwork.Clone();
                var clonedGame = new Game(headless: true) {
                    snake = {
                        body = new List<Vector2>(Game.snake.body),
                        direction = Game.snake.direction,
                        isAddSegment = Game.snake.isAddSegment
                    },
                    score = Game.score,
                    cachedScore = Game.cachedScore,
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

                // 🎯 === MAJOR REWARDS BONUSES ===

                // 🍎 Base reward for eating
                fitness += Game.cachedScore * 100;

                // 🚀 Boost based on distance
                if(Game.startDistanceToFood > 0 && Score > 0) {
                    fitness += Game.startDistanceToFood * Score * 10;
                }

                // ⏱️ Reward for surviving
                fitness += Game.stepsSurvived * 0.05;

                // 📏 Bonus for snake length
                fitness += Math.Pow(SnakeLength, 2.2);

                // EFFICIENCY BONUS (reward fast eating)
                if(Score > 0) {
                    double avgStepsPerFood = (double)Game.stepsSurvived / Score;
                    double efficiencyBonus = 50.0 / Math.Max(1.0, avgStepsPerFood / 50.0);
                    fitness += efficiencyBonus;
                }

                // DISTANCE-BASED REWARD (reward reducing distance to food)
                if(Game.startDistanceToFood > 0 && Score == 0) {
                    double distanceReduction = Game.startDistanceToFood - distanceToApple;
                    fitness += distanceReduction * 2.0; 
                }

                // 🧠 === STRATEGIC BONUSES ===

                double stepsLeft = Game.maxStepsWithoutFood - Game.stepsSinceLastFood;
                double stepsLeftNormalized = stepsLeft / (double)Game.maxStepsWithoutFood;
                double eatReward = 1.0 + (stepsLeftNormalized * 0.5);
                fitness += eatReward;
                
                // Direction alignment bonus
                float dx = applePosition.X - headPosition.X;
                float dy = applePosition.Y - headPosition.Y;

                if((dx > 0 && Game.snake.direction.X == 1) || (dx < 0 && Game.snake.direction.X == -1))
                    fitness += 0.5;  // Reduced from 1.0
                if((dy > 0 && Game.snake.direction.Y == 1) || (dy < 0 && Game.snake.direction.Y == -1))
                    fitness += 0.5;  // Reduced from 1.0

                // ❌ PENALTIES
                // Distance penalty (mild)
                fitness -= distanceToApple * 0.02;

                // Collision penalties
                if(Game.causeOfDeath == Game.DeathCause.CollidedToBody)
                    fitness -= 100;  
                if(Game.causeOfDeath == Game.DeathCause.CollidedToWall)
                    fitness -= 30;

                // Zero score penalty (failed to find food)
                if(Score == 0 && IsSnakeDead)
                    fitness *= 0.2;

                // Ensure non-negative fitness
                Fitness = Math.Max(0, fitness);
                HasFinalFitness = true;

                Fitness = fitness;
            }

            public void Step(Game game, int move) {
                if(!game.isRunning)
                    return;

                Matrix<double> input = Matrix<double>.Build.Dense(1, game.GetInputs().Length, (i, j) => game.GetInputs()[j]);
                (_, move) = NeuralNetwork.Predict(input);

                Vector2 currentDir = game.snake.direction;
                Vector2 left = new Vector2(-currentDir.Y, currentDir.X);
                Vector2 right = new Vector2(currentDir.Y, -currentDir.X);

                switch(move) {
                case 0:
                    break;
                case 1:
                    game.snake.direction = left;
                    break;
                case 2:
                    game.snake.direction = right;
                    break;
                }

                game.Update();
                game.Draw();
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
                body = new List<Vector2> {
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
            public int cachedScore = 0;
            public int stepCap = 500;
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
                if(!isHeadless) {
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
                    causeOfDeath = DeathCause.CollidedToWall;
                }
            }

            // ⚡ OPTIMIZATION: Early exit collision detection, no list allocation
            public void CheckCollisionWithTail() {
                Vector2 head = snake.body[0];
                for(int i = 1; i < snake.body.Count; i++) {
                    if(snake.body[i] == head) {
                        GameOver();
                        PlaySoundSafe(wallSound);
                        causeOfDeath = DeathCause.CollidedToBody;
                        return; // Early exit
                    }
                }
            }

            // ⚡ OPTIMIZATION: Inline collision check
            private bool IsCollision(Vector2 position) {
                if(position.X < 0 || position.Y < 0 || position.X >= cellCount || position.Y >= cellCount)
                    return true;

                // Check self-collision with early exit
                for(int i = 1; i < snake.body.Count; i++) {
                    if(snake.body[i] == position)
                        return true;
                }
                return false;
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

                    int stepsLimit = Math.Min(maxStepsWithoutFood +  10, stepCap);
                    CheckCollisionWithFood();
                    CheckCollisionWithEdges();

                    if(!isGhostMode)
                        CheckCollisionWithTail();

                    if(Raylib.IsKeyPressed(KeyboardKey.K) || (stepsSinceLastFood > stepsLimit)) 
                        GameOver();

                    if(Raylib.IsKeyPressed(KeyboardKey.R)) 
                        Reset();
                }
            }

            public void GameOver() {
                snake.Reset();
                food.position = GenerateRandomPos(snake.body);
                isRunning = false;
                stepsSinceLastFood = 0;
                cachedScore = score;
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

                Vector2 head = snake.body[0];
                Vector2 dir = snake.direction;
                Vector2 left = new Vector2(-dir.Y, dir.X);
                Vector2 right = new Vector2(dir.Y, -dir.X);

                // Ray-based vision: 8 directions × 3 object types
                Vector2[] directions = new Vector2[] {
                    new Vector2(0, -1),   // Up
                    new Vector2(1, -1),   // Up-Right
                    new Vector2(1, 0),    // Right
                    new Vector2(1, 1),    // Down-Right
                    new Vector2(0, 1),    // Down
                    new Vector2(-1, 1),   // Down-Left
                    new Vector2(-1, 0),   // Left
                    new Vector2(-1, -1)   // Up-Left
                };

                const int visionRange = 10;
                foreach(var dirVec in directions) {
                    bool seenFood = false, seenBody = false, seenWall = false;

                    for(int dist = 1; dist <= visionRange; dist++) {
                        Vector2 pos = head + dirVec * dist;

                        if(!IsInsideGrid(pos)) {
                            seenWall = true;
                            break;
                        }

                        if(!seenFood && food.position == pos)
                            seenFood = true;
                        if(!seenBody && snake.body.Contains(pos))
                            seenBody = true;
                        if(seenFood && seenBody)
                            break;
                    }

                    inputs.Add(seenFood ? 1.0 : 0.0);
                    inputs.Add(seenBody ? 1.0 : 0.0);
                    inputs.Add(seenWall ? 1.0 : 0.0);
                }

                // Direction
                inputs.Add(dir.X);
                inputs.Add(dir.Y);

                // Danger sensors
                inputs.Add(IsCollision(head + dir) ? 1.0 : 0.0);
                inputs.Add(IsCollision(head + left) ? 1.0 : 0.0);
                inputs.Add(IsCollision(head + right) ? 1.0 : 0.0);

                // Relative food direction
                Vector2 toFood = Vector2.Normalize(food.position - head);
                inputs.Add(Vector2.Dot(dir, toFood));
                inputs.Add(Vector2.Dot(left, toFood));
                inputs.Add(Vector2.Dot(right, toFood));

                // Normalized snake length
                inputs.Add((double)snake.body.Count / (cellCount * cellCount));

                // Global food direction
                inputs.Add(food.position.Y < head.Y ? 1.0 : 0.0);  // Food up
                inputs.Add(food.position.Y > head.Y ? 1.0 : 0.0);  // Food down
                inputs.Add(food.position.X < head.X ? 1.0 : 0.0);  // Food left
                inputs.Add(food.position.X > head.X ? 1.0 : 0.0);  // Food right

                
                return inputs.ToArray();
            }
        }
        public static bool IsElementInDeque(Vector2 element, List<Vector2> deque) {
            return deque.Contains(element);
        }
    }
}