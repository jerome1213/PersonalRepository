package sort.sort;

import java.util.ArrayList;
import java.util.List;

public class MergeSort extends SortMain implements IFunctions {
    // https://www.youtube.com/watch?v=FCAtxryNgq4&list=PLLcbGhhl4sQDIp8j8L-OuI9n7oOuEGrnJ&index=11
    // https://gmlwjd9405.github.io/2018/05/08/algorithm-merge-sort.html

    /*
    1. 분할과정과 정복과정으로 나우어짐
    2. 모든 숫자를 독립적으로 분할
    3. 그룹별로 엘레멘트의 크기를 비교하며 하나의 그룹으로 병합
    4. Time Complexity
        - Worst     : O (n log n)
        - Average   : O (n log n)
        - Best      : O (n log n)
    */

    protected ArrayList<Integer> testRandomList;
    int[] sorted = null;

    public MergeSort() {
        testRandomList = (ArrayList<Integer>) mainTestRandomList.clone();
        System.out.println(this.getClass().getName());
    }

    @Override
    public void runSort() {
        beforeTime = System.currentTimeMillis();

        sorted = new int[testRandomList.size()];
        mergeSort(testRandomList, 0, testRandomList.size() - 1);

        printDiffTime();
    }

    private void mergeSort(List<Integer> arr, int start, int end) {
        if (start < end) {
            int center = getCenterPosition(start, end);
            mergeSort(arr, start, center);
            mergeSort(arr, center + 1, end);

            merge(arr, start, end);
        }
    }

    private void merge(List<Integer> arr, int start, int end) {
        int center = getCenterPosition(start, end);
        int leftArrPosition = start;
        int rightArrPosition = center + 1;
        int sortedArrPosition = start;

        while (leftArrPosition <= center && rightArrPosition <= end) {
            if (arr.get(leftArrPosition) < arr.get(rightArrPosition)) {
                sorted[sortedArrPosition++] = arr.get(leftArrPosition++);
            } else {
                sorted[sortedArrPosition++] = arr.get(rightArrPosition++);
            }
        }

        while (leftArrPosition <= center) {
            sorted[sortedArrPosition++] = arr.get(leftArrPosition++);
        }

        while (rightArrPosition <= end) {
            sorted[sortedArrPosition++] = arr.get(rightArrPosition++);
        }

        for (int i = start; i <= end; i++) {
            arr.set(i, sorted[i]);
        }
    }

    public void printDiffTime() {
        System.out.println(testRandomList);
        super.printDiffTime();
    }
}
