package sort.sort;

public class HeapSort2 {
    public static void main(String[] args) {
        int[] input = new int[]{10, 26, 5, 37, 1, 61, 11, 59, 15, 48, 19};

        HeapSort2 heapSort2 = new HeapSort2();
        heapSort2.heapSort(input);
    }

    private void heapSort(int[] input) {
        // make complete binary tree
        for (int i = 1; i < input.length; i++) {
            int newNode = i;
            do {
                int parentIdx = (newNode-1) / 2;
                if (input[newNode] > input[parentIdx]) {
                    swap(input, newNode, parentIdx);
                }
                newNode = parentIdx;
            } while (newNode > 0);
        }
        printStatus(input);

        // sort
        for (int i = input.length - 1; i > 0; i--) {
            int root = 0;
            int childIdx;
            swap(input, i, root);
            do {
                childIdx = (root * 2) + 1;
                if (childIdx + 1 < i && input[childIdx] < input[childIdx + 1]) {
                    childIdx++;
                }
                if (childIdx < i && input[root] < input[childIdx]) {
                    swap(input, root, childIdx);
                }
                root = childIdx;
            } while (childIdx < i);
            printStatus(input);
        }
    }

    private void swap(int[] array, int x, int y) {
        int temp = array[x];
        array[x] = array[y];
        array[y] = temp;
    }

    private void printStatus(int[] arr) {
        for (int i = 0; i < arr.length; i++) {
            System.out.print(arr[i] + " ");
        }
        System.out.println();
    }
}
