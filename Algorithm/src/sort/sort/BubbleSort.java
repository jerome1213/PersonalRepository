package sort.sort;

import java.util.ArrayList;

public class BubbleSort extends SortMain implements IFunctions {

    // https://gmlwjd9405.github.io/2018/05/06/algorithm-bubble-sort.html

    protected ArrayList<Integer> testRandomList;

    public BubbleSort() {
        testRandomList = (ArrayList<Integer>) mainTestRandomList.clone();
        System.out.println(this.getClass().getName());
    }

    @Override
    public void runSort() {
        beforeTime = System.currentTimeMillis();

        for (int i = 0; i < numRandomInteger - 1; i++) {
            for (int j = 0; j < numRandomInteger - 1 - i; j++) {
                if (testRandomList.get(j) > testRandomList.get(j + 1)) {
                    swap(testRandomList, j, j + 1);
                }
            }
        }

        printDiffTime();
    }

    public void printDiffTime() {
        System.out.println(testRandomList);
        super.printDiffTime();
    }
}
