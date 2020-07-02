package sort.datastructure;

public class Heap {
    public static void main(String[] args) {
        int[] input = new int[]{7, 6, 5, 8, 3, 5, 9, 1, 6};

        Heap heap = new Heap();
        int[] result = heap.constructHeap(input);

        for (int node : result) {
            System.out.print(node + " ");
        }
        System.out.println();

        heap.sort(result);
        for (int node : result) {
            System.out.print(node + " ");
        }
    }

    private int[] constructHeap(int[] array) {
        for (int i = 1; i < array.length; i++) {
            int c = i;
            do {
                int parent = (c - 1) / 2;
                if (array[parent] < array[c]) {
                    swap(array, parent, c);
                }
                c = parent;
            } while (c != 0);
        }
        return array;
    }

    private void swap(int[] array, int x, int y) {
        int temp = array[x];
        array[x] = array[y];
        array[y] = temp;
    }

    private void sort(int[] array) {
        for (int i = array.length - 1; i >= 0; i--) {
            int parent = 0;
            swap(array, i, parent);
            int leftChild = 0;
            do {
                leftChild = parent * 2 + 1;
                if (leftChild < i - 1 && array[leftChild] < array[leftChild + 1]) {
                    leftChild++;
                }
                if (leftChild < i && array[parent] < array[leftChild]) {
                    swap(array, parent, leftChild);
                }
                parent = leftChild;
            } while (i > leftChild);
        }
    }
}
