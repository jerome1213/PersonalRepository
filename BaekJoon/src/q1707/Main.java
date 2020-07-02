package q1707;

import algorithm.algo127.Solution;
import algorithm.algo127.Solution2;

import java.util.Arrays;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        String beginWord = "a";
        String endWord = "c";
        List<String> wordList = Arrays.asList(new String[]{"a", "b", "c"});

        Solution solution = new Solution();
        solution.ladderLength(beginWord, endWord, wordList);
    }
}
