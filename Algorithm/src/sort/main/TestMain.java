package sort.main;

import sort.sort.*;

public class TestMain {
    public static void main(String[] args) {
        BubbleSort bubbleSort = new BubbleSort();
        bubbleSort.runSort();

        SelectionSort selectionSort = new SelectionSort();
        selectionSort.runSort();

        InsertionSort insertionSort = new InsertionSort();
        insertionSort.runSort();

        MergeSort mergeSort = new MergeSort();
        mergeSort.runSort();

        QuickSort quickSort = new QuickSort();
        quickSort.runSort();
    }
}
