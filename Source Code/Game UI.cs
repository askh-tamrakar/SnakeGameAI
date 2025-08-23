using Raylib_cs;
using System;
using System.Collections;
using System.Numerics;
using static SnakeGameAI.Snake_Game;
using static SnakeGameAI.Program;
using Color = Raylib_cs.Color;
using Rectangle = Raylib_cs.Rectangle;

namespace SnakeGameAI { 
    public class Game_UI {
        const int offset = 25;
        const int lineThickness = 4;

        static int debugLogWidth = (int)((windowDim.X - (cellCount * cellSize + 2 * offset)) - lineThickness);
        static int debugLogHeight = (int)(windowDim.Y - (2 * offset) - (2 * 10));

        private float scrollOffset = 0;
        private int selectedIndex = -1; // Persistent selection

        public void DrawScrollableBestGenomeList(List<Genome> bestGenomes) {
            float itemHeight = 50f;

            int startPosY = offset + Program.offset;
            int startPosX = 982 + offset;
            int width = debugLogWidth - ((2 * offset) + Program.offset);
            int height = 400;

            Vector2 position = new(startPosX, startPosY);
            Rectangle scrollRect = new Rectangle(position.X, position.Y, width, height);

            // Scroll Wheel Input
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
                int score = g.IsSnakeDead ? g.Game.updateScore : g.Score ;
                string snake = g.IsSnakeDead ? "Snake is Dead" : " Snake is Alive";

                string text = $"ID: {g.GenomeID} | Score: {score}," +
                              $" Fitness: {g.Fitness:F2}, " +
                              $"{snake}, " +
                              $"Steps Survived: {g.StepsSnapshot}";

                Rectangle itemRect = new Rectangle(position.X, y, width, itemHeight);
                Vector2 mousePos = Raylib.GetMousePosition();

                bool isHovered = Raylib.CheckCollisionPointRec(mousePos, itemRect);
                bool isSelected = selectedIndex == i;

                Color bgColor = isSelected ? AppColors.YellowHighlight.Ray :
                                isHovered ? AppColors.JungleGreen.Ray : AppColors.ForestGreen.Ray;

                Color textColor = isSelected ? AppColors.ForestGreen.Ray :
                                  isHovered ?  AppColors.YellowHighlight.Ray : AppColors.YellowPastel.Ray;
                Raylib.DrawRectangleRec(itemRect, bgColor);

                // Mouse Click Selection
                if(isHovered) {
                    if(Raylib.IsMouseButtonPressed(MouseButton.Left)) {
                        selectedIndex = i;
                        population.FinalGenomeFromList = g;
                    }
                    
                    if (Raylib.IsKeyPressed(KeyboardKey.Delete)) {
                        population.BestEverGenomeList.Remove(g);
                    }
                }
                Raylib.DrawTextEx(font, text, new Vector2(position.X + 10, y + 12.5f), 28, 1, textColor);
            }

            Raylib.EndScissorMode();
        }
    }
}
