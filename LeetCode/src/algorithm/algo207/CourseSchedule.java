package algorithm.algo207;

import java.util.*;

public class CourseSchedule {

    public static void main(String[] args) {
        int numCourses = 3;
        int[][] prerequisites = {{2, 0}, {2, 1}};

        Solution solution = new Solution();
        solution.canFinish(numCourses, prerequisites);
    }
}

class Solution {
    // 1. indegree == 0 인 것들 탐색
    // 2. 관련 edge 제거, indegree 업데이트
    // 1,2 반복
    // 루프 종료 후, 모든 노드 탐색 했는지 확인

    public boolean canFinish(int numCourses, int[][] prerequisites) {

        List<List<Integer>> adjList = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) {
            adjList.add(new ArrayList<Integer>());
        }

        int[] indegree = new int[numCourses];
        for (int[] edge : prerequisites) {
            int node = edge[0];
            indegree[node]++;
            adjList.get(edge[1]).add(node);
        }

        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0) {
                queue.offer(i);
            }
        }

        Set<Integer> visited = new HashSet<>();

        while (!queue.isEmpty()) {
            int node = queue.poll();
            visited.add(node);

            for (int dest : adjList.get(node)) {
                if (--indegree[dest] == 0) {
                    queue.offer(dest);
                }
            }
            adjList.get(node).clear();
        }

        return visited.size() == numCourses;
    }
}
