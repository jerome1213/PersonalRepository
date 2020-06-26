package sort.main;

import sort.sort.BubbleSort;
import sort.sort.SelectionSort;

public class TestMain {
    public static void main(String[] args) {
        BubbleSort bubbleSort = new BubbleSort();
        bubbleSort.runSort();

        SelectionSort selectionSort = new SelectionSort();
        selectionSort.runSort();

    }
}
