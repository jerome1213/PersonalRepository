package algorithm.algo127;

import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class Solution2 {
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        Queue<String> q = new LinkedList<>();
        boolean noEndWord = true;
        for (String word : wordList) {
            if (word.equals(endWord)) noEndWord = false;
            q.offer(word);
        }

        if (noEndWord) return 0;

        String compare = beginWord;
        System.out.print(beginWord);

        int wordLen = beginWord.length();
        int ladderLength = 0;
        while (!q.isEmpty()) {
            String word = q.poll();
            int score = 0;
            for (int i = 0; i < wordLen; i++) {
                if (word.charAt(i) == compare.charAt(i)) score++;
            }
            if (score == wordLen - 1) {
                ladderLength++;
                compare = word;
                System.out.print(" -> " + word);
                if (word.equals(endWord)) break;
            } else if (q.size()>1) {
                q.offer(word);
            }
        }

        return ladderLength;
    }
}
