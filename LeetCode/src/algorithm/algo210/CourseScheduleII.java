package algorithm.algo210;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Queue;

public class CourseScheduleII {

    public static void main(String[] args) {
        int numCourses = 2;
        // [[0,1],[0,2],[1,3],[3,0]]
        int[][] prerequisites = {{1, 0}};

        Solution solution = new Solution();
        solution.findOrder(numCourses, prerequisites);
    }
}

class Solution {

    // using BFS

    public int[] findOrder(int numCourses, int[][] prerequisites) {
        int[] indegree = new int[numCourses];
        int[] topologicalOutput = new int[numCourses];

        HashMap<Integer, ArrayList<Integer>> adjList = new HashMap<>();

        for (int[] edge : prerequisites) {
            ArrayList<Integer> list = adjList.getOrDefault(edge[1], new ArrayList<>());
            list.add(edge[0]);
            adjList.put(edge[1], list);
            indegree[edge[0]]++;
        }

        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0) {
                queue.offer(i);
            }
        }

        int step = 0;
        while (!queue.isEmpty()) {
            int node = queue.poll();
            topologicalOutput[step++] = node;

            if (adjList.containsKey(node)) {
                for (int next : adjList.get(node)) {
                    if (--indegree[next] == 0) {
                        queue.offer(next);
                    }
                }
            }
        }

        return (step == numCourses) ? topologicalOutput : new int[0];
    }
}