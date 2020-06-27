package search.search;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.PriorityQueue;
import java.util.Queue;

public class BreadthFirstSearch {
    // BFS, Breadth-First Search
    // https://gmlwjd9405.github.io/2018/08/15/algorithm-bfs.html

    public static void main(String[] args) {

        Graph graph = new Graph(8);

        graph.addEdge(0, 1);
        graph.addEdge(0, 2);
        graph.addEdge(0, 4);
        graph.addEdge(1, 0);
        graph.addEdge(1, 2);
        graph.addEdge(2, 0);
        graph.addEdge(2, 1);
        graph.addEdge(2, 3);
        graph.addEdge(2, 4);
        graph.addEdge(3, 2);
        graph.addEdge(3, 4);
        graph.addEdge(4, 0);
        graph.addEdge(4, 2);
        graph.addEdge(4, 3);

        graph.breadthFirstSearch(0);

    }

}

class Graph {
    private ArrayList<Integer>[] adj;
    private boolean[] visited;
    private Queue<Integer> queue;

    public Graph(int numOfNode) {
        this.visited = new boolean[numOfNode];
        this.queue = new PriorityQueue<>();

        adj = new ArrayList[numOfNode];
        for (int i = 0; i < numOfNode; ++i) {
            adj[i] = new ArrayList<>();
        }
    }

    public void addEdge(int v, int w) {
        adj[v].add(w);
    }

    public void breadthFirstSearch(int startNode) {
        queue.add(startNode);
        visited[startNode] = true;
        System.out.print(startNode + "  ");

        while (!queue.isEmpty()) {
            int next = queue.poll();
            addQueue(adj[next]);
        }

    }

    private void addQueue(ArrayList<Integer> linkedList) {
        for (int i = 0; i < linkedList.size(); i++) {
            int next = linkedList.get(i);
            if (!visited[next]) {
                queue.add(next);
                visited[next] = true;
                System.out.print(next + "  ");
            }
        }
    }

}