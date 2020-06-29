package bfs.q2178;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Scanner;

public class MazeExploration {

    private static int N, M;
    private static int[][] array;
    private static boolean[][] visited;

    private static final int[] dx = {0, 1, 0, -1};
    private static final int[] dy = {1, 0, -1, 0};

    public static void main(String[] args) throws FileNotFoundException {
        Scanner scanner = new Scanner(new FileInputStream("./BaekJoon/src/bfs/q2178/input"));

        N = scanner.nextInt();
        M = scanner.nextInt();
        scanner.nextLine();

        array = new int[N][M];
        visited = new boolean[N][M];

        for (int i = 0; i < N; i++) {
            String line = scanner.nextLine();
            for (int j = 0; j < M; j++) {
                array[i][j] = Integer.parseInt(String.valueOf(line.charAt(j)));
            }
        }

        visited[0][0] = true;
        bfs(0, 0);

        System.out.println(array[N - 1][M - 1]);
    }

    private static void bfs(int x, int y) {
        Queue<Dot> queue = new LinkedList<>();
        queue.add(new Dot(x, y));

        while (!queue.isEmpty()) {
            Dot dot = queue.poll();
            for (int i = 0; i < 4; i++) {
                int nextX = dot.x + dx[i];
                int nextY = dot.y + dy[i];

                if ((nextX < 0 || nextY < 0 || nextX >= N || nextY >= M)
                        || (visited[nextX][nextY] || array[nextX][nextY] == 0)) {
                    continue;
                }

                queue.add(new Dot(nextX, nextY));
                visited[nextX][nextY] = true;
                array[nextX][nextY] = array[dot.x][dot.y] + 1;
            }
        }
    }
}

class Dot {
    int x;
    int y;

    Dot(int x, int y) {
        this.x = x;
        this.y = y;
    }
}
