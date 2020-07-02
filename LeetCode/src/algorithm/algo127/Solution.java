package algorithm.algo127;


import java.util.*;

public class Solution {
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        boolean noEndWord = true;
        for (String word : wordList) {
            if (word.equals(endWord)) {
                noEndWord = false;
                break;
            }
        }
        if (noEndWord) return 0;
        return dfs(beginWord, endWord, wordList);
    }

    public int bfs(String beginWord, String endWord, List<String> wordList) {
        boolean noEndWord = true;
        for (String word : wordList) {
            if (word.equals(endWord)) {
                noEndWord = false;
                break;
            }
        }
        if (noEndWord) return 0;

        Set<String> visited = new HashSet<>();
        int wordLen = beginWord.length();
        Queue<String> q = new LinkedList<>();
        q.offer(beginWord);

        int ladderLength = 0;
        while (!q.isEmpty()) {
            int qSize = q.size();
            boolean hasNext = false;
            for (int i = 0; i < qSize; i++) {
                String compare = q.poll();
                if (compare == null) continue;

                if (endWord.equals(compare)) return ladderLength + 1;

                for (String word : wordList) {
                    int score = 0;
                    for (int j = 0; j < wordLen; j++) {
                        if (compare.charAt(j) == word.charAt(j)) score++;
                    }
                    if (score == wordLen - 1 && !visited.contains(word)) {
                        hasNext = true;
                        q.offer(word);
                        visited.add(word);
                    }
                }
            }
            if (hasNext) ladderLength++;
        }
        return 0;
    }

    private int dfs(String compare, String endWord, List<String> wordList) {
        ArrayList<String> list = new ArrayList<>(wordList);
        int wordLen = compare.length();
        for (int i = 0; i < list.size(); i++) {
            String word = wordList.get(i);
            int depth = 0;
            int score = 0;
            if (word.equals(compare)) {
                list.remove(i);
                continue;
            }
            for (int j = 0; j < wordLen; j++) {
                if (compare.charAt(j) == word.charAt(j)) score++;
            }
            if (score == wordLen - 1) {
                list.remove(i);
                if (word.equals(endWord)) {
                    return depth + 1;
                } else {
                    depth = dfs(word, endWord, list) + 1;
                    return depth;
                }
            }
        }
        return 0;
    }
}