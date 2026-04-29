# FP-GROWTH ALGORITHM (based on tree-growth strategy)
# %%
import numpy as np
import pandas as pd
from collections import Counter

# %%
class FPNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.next = None

    def increment(self, count):
        self.count += count

# %%
def _update_header_table(item, node, header_table):
    if header_table[item][1] is None:
        header_table[item][1] = node
        return

    current = header_table[item][1]
    while current.next is not None:
        current = current.next
    current.next = node

# %%
def _insert_tree(items, node, header_table):
    if not items:
        return

    first_item = items[0]
    if first_item in node.children:
        node.children[first_item].increment(1)
    else:
        new_child = FPNode(first_item, 1, node)
        node.children[first_item] = new_child
        _update_header_table(first_item, new_child, header_table)

    _insert_tree(items[1:], node.children[first_item], header_table)

# %%
def build_fp_tree(transactions, min_support_count):
    item_counter = Counter()
    for transaction in transactions:
        item_counter.update(transaction)

    frequent_items = {
        item: count
        for item, count in item_counter.items()
        if count >= min_support_count
    }
    if not frequent_items:
        return None, None

    # Header table format: {item: [support_count, first_node_in_linked_list]}
    header_table = {
        item: [count, None] for item, count in frequent_items.items()
    }
    root = FPNode(None, 0, None)

    for transaction in transactions:
        filtered = [item for item in transaction if item in frequent_items]
        filtered.sort(key=lambda item: (frequent_items[item], item), reverse=True)
        if filtered:
            _insert_tree(filtered, root, header_table)

    return root, header_table

# %%
def _ascend_tree(node):
    path = []
    while node.parent is not None and node.parent.item is not None:
        node = node.parent
        path.append(node.item)
    return path

# %%
def _conditional_pattern_base(item, header_table):
    bases = []
    node = header_table[item][1]

    while node is not None:
        prefix_path = _ascend_tree(node)
        if prefix_path:
            bases.append((prefix_path, node.count))
        node = node.next

    return bases

# %%
def mine_fp_tree(header_table, min_support_count, prefix, frequent_itemsets):
    # Mine from least frequent item to most frequent item
    sorted_items = sorted(header_table.items(), key=lambda x: (x[1][0], x[0]))

    for item, (support_count, _) in sorted_items:
        new_pattern = prefix.union({item})
        frequent_itemsets[frozenset(new_pattern)] = support_count

        conditional_bases = _conditional_pattern_base(item, header_table)
        conditional_transactions = []
        for path, count in conditional_bases:
            conditional_transactions.extend([path] * count)

        _, conditional_header = build_fp_tree(
            conditional_transactions,
            min_support_count=min_support_count,
        )

        if conditional_header is not None:
            mine_fp_tree(
                conditional_header,
                min_support_count=min_support_count,
                prefix=new_pattern,
                frequent_itemsets=frequent_itemsets,
            )

# %%
def fpgrowth(transactions, min_support=0.04):
    if not 0 < min_support <= 1:
        raise ValueError("min_support must be in the interval (0, 1].")

    min_support_count = int(np.ceil(min_support * len(transactions)))
    _, header_table = build_fp_tree(transactions, min_support_count=min_support_count)

    if header_table is None:
        return pd.DataFrame(columns=["support", "itemsets"]) 

    frequent_itemsets = {}
    mine_fp_tree(
        header_table,
        min_support_count=min_support_count,
        prefix=set(),
        frequent_itemsets=frequent_itemsets,
    )

    result = pd.DataFrame(
        [
            {
                "support": count / len(transactions),
                "itemsets": itemset,
            }
            for itemset, count in frequent_itemsets.items()
        ]
    )

    result = result.sort_values(["support", "itemsets"], ascending=[False, True]).reset_index(drop=True)
    return result

# %%
from data.import_data import useProductsTransactionsDataset

products_transactions = useProductsTransactionsDataset()
fp_growth_result = fpgrowth(products_transactions.tolist(), min_support=0.04)
fp_growth_result.head(20)