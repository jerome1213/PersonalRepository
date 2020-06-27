package sort.main;

import sort.sort.BubbleSort;
import sort.sort.InsertionSort;
import sort.sort.MergeSort;
import sort.sort.SelectionSort;

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

    }
}
