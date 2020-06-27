package sort.sort;

import java.util.ArrayList;

public class QuickSort extends SortMain implements IFunctions {

    /*
    https://www.youtube.com/watch?v=cWH49IKDIiI&list=PLLcbGhhl4sQDIp8j8L-OuI9n7oOuEGrnJ&index=12
    https://www.daleseo.com/sort-quick/

    1. 분할과정과 정복과정으로 나누어짐
    2. 피봇을 정한 뒤 외쪽 퀵소트, 오른쪽 퀵소트
    3. 다양한 피봇 선정 방식 -> 다양한 퀵소트
    4. Time Complexity
        - Worst     : O (n2)
        - Average   : O (n log n)
        - Best      : O (n log n)
    */

    protected ArrayList<Integer> testRandomList;

    public QuickSort() {
        testRandomList = (ArrayList<Integer>) mainTestRandomList.clone();
        System.out.println(this.getClass().getName());
    }

    @Override
    public void runSort() {
        beforeTime = System.currentTimeMillis();

        quickSort(testRandomList, 0, testRandomList.size() - 1);

        printDiffTime();
    }

    private void quickSort(ArrayList<Integer> arr, int start, int end) {
        if (start < end) {
            int partition = partition(arr, start, end);

            quickSort(arr, start, partition - 1);
            quickSort(arr, partition, end);
        }
    }

    private int partition(ArrayList<Integer> arr, int start, int end) {
        int pivot = arr.get(getCenterPosition(start, end));

        while (start <= end) {
            while (arr.get(start) < pivot) start++;
            while (arr.get(end) > pivot) end--;
            if (start <= end) {
                swap(arr, start, end);
                start++;
                end--;
            }
        }
        return start;
    }

    public void printDiffTime() {
        System.out.println(testRandomList);
        super.printDiffTime();
    }
}
