package algorithm.algo111;

import sun.reflect.generics.tree.Tree;

public class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode() {
    }

    TreeNode(int val) {
        this.val = val;
    }

    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }

    TreeNode(int[] array) {
        TreeNode node = makeTree(array, new TreeNode(), 0);
        this.val = node.val;
        this.left = node.left;
        this.right = node.right;
    }

    TreeNode makeTree(int[] arr, TreeNode root, int i) {
        if (i < arr.length) {
            if (arr[i] >= 0) {
                TreeNode node = new TreeNode(arr[i]);
                root = node;
                root.left = makeTree(arr, root.left, 2 * i + 1);
                root.right = makeTree(arr, root.right, 2 * i + 2);
            }
        }
        return root;
    }
}
