package algorithm.algo111;

import java.util.LinkedList;
import java.util.Queue;

public class MinimumDepthOfBinaryTree {
    public static void main(String[] args) {
        TreeNode node = new TreeNode(new int[]{3, 9, 20, -1, -1, 15, 7});
        Solution solution = new Solution();
        solution.minDepth(node);
    }
}

class Solution {

    public int minDepth(TreeNode root) {
        int depth = 0;
        Queue<TreeNode> q = new LinkedList<>();

        q.offer(root);

        while (!q.isEmpty()) {
            int sizeOfQ = q.size();
            for (int i = 0; i < sizeOfQ; i++) {
                TreeNode node = q.poll();
                if (node == null) {
                    continue;
                }

                if(i==0) depth++;

                if (node.left == null && node.right == null) {
                    return depth;
                }
                if (node.left != null) {
                    q.offer(node.left);
                }
                if (node.right != null) {
                    q.offer(node.right);
                }
            }
        }

        return depth;
    }
}