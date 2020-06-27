package sort.sort;

import java.util.ArrayList;

public class SelectionSort extends SortMain implements IFunctions {
    protected ArrayList<Integer> testRandomList;

    public SelectionSort() {
        testRandomList = (ArrayList<Integer>) mainTestRandomList.clone();
        System.out.println(this.getClass().getName());
    }

    @Override
    public void runSort() {
        beforeTime = System.currentTimeMillis();

        for (int i = 0; i < numRandomInteger - 1; i++) {
            int min = i;
            for (int j = i + 1; j < numRandomInteger; j++) {
                if (testRandomList.get(min) > testRandomList.get(j)) {
                    min = j;
                }
            }
            swap(testRandomList, min, i);
        }

        printDiffTime();
    }
}
