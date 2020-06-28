package sort.sort;

import java.util.ArrayList;

public class InsertionSort extends SortMain implements IFunctions {
    // https://www.youtube.com/watch?v=iqf96rVQ8fY

    protected ArrayList<Integer> testRandomList;
    public InsertionSort() {
        testRandomList = (ArrayList<Integer>) mainTestRandomList.clone();
        System.out.println(this.getClass().getName());
    }

    @Override
    public void runSort() {
        beforeTime = System.currentTimeMillis();

        // jerome
        for (int i = 0; i < numRandomInteger - 1; i++) {
            int key = i + 1;
            for (int j = i; j >= 0; j--) {
                if (testRandomList.get(j) > testRandomList.get(key)) {
                    swap(testRandomList, j, key);
                    key = j;
                } else {
                    break;
                }
            }
        }

        // web
        /*for (int i = 1; i < numRandomInteger; i++) {
            int key = testRandomList.get(i);
            int j = i - 1;
            while (j >= 0 && testRandomList.get(j) > key) {
                swap(testRandomList, j + 1, j);
                j--;
            }
        }*/

        printDiffTime();
    }

    public void printDiffTime() {
        System.out.println(testRandomList);
        super.printDiffTime();
    }
}
