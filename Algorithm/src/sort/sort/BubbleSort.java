package sort.sort;

public class BubbleSort extends SortMain implements IFunctions {

    // https://gmlwjd9405.github.io/2018/05/06/algorithm-bubble-sort.html


    public BubbleSort() {
        System.out.println("BubbleSort");
    }

    @Override
    public void runSort() {
        beforeTime = System.currentTimeMillis();

        for (int i = 0; i < numRandomInteger - 1; i++) {
            for (int j = 0; j < numRandomInteger - 1 - i; j++) {
                if (testRandomList.get(j) > testRandomList.get(j + 1)) {
                    swap(j, j + 1);
                }
            }
        }

        printDiffTime();
    }
}
