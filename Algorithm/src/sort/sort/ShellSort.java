package sort.sort;

import java.util.ArrayList;

public class ShellSort extends SortMain implements IFunctions {

    // https://gmlwjd9405.github.io/2018/05/08/algorithm-shell-sort.html


    protected ArrayList<Integer> testRandomList;

    public ShellSort() {
        testRandomList = (ArrayList<Integer>) mainTestRandomList.clone();
        System.out.println(this.getClass().getName());
    }

    @Override
    public void runSort() {
        beforeTime = System.currentTimeMillis();

        shellSort(testRandomList, 0, testRandomList.size() - 1);

        printDiffTime();
    }

    private void shellSort(ArrayList<Integer> arr, int left, int right) {
        int interval = (left + right) / 2;

        while (interval > 0) {

            for (int i = left; i < numRandomInteger; i += interval) {
                int key = testRandomList.get(i);
                int j = i - interval;
                while (j >= 0 && testRandomList.get(j) > key) {
                    swap(testRandomList, j + interval, j);
                    j -= interval;
                }
            }

            interval /= 2;
        }
    }

    @Override
    public void printDiffTime() {
        System.out.println(testRandomList);
        super.printDiffTime();
    }
}
