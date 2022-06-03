import sys
import os

from nltk.parse.corenlp import CoreNLPDependencyParser

dep_parser = CoreNLPDependencyParser(url="http://localhost:9000")

DEBUG = os.environ.get("DEBUG", False)


class deptree:

    # --------------------------------------------------------------
    # analyze a sentence with stanforCore and get a dependency tree
    def __init__(self, txt):
        "analyze a sentence with stanforCore and get a dependency tree"

        if txt == "":
            self.tree = None
        else:
            txt2 = (
                txt.replace("/", " / ")
                .replace("-", " - ")
                .replace(".", ". ")
                .replace("'", " ' ")
            )
            (self.tree,) = dep_parser.raw_parse(txt2)
            offset = 0

            if DEBUG:
                print(txt2, file=sys.stderr)
                self.print(file=sys.stderr)

            tks = []
            for t in self.get_nodes():
                # enrich tree nodes with offset in original text.
                word = self.tree.nodes[t]["word"]
                offset = txt.find(word, offset)
                self.tree.nodes[t]["start"] = offset
                self.tree.nodes[t]["end"] = offset + len(word) - 1
                offset += len(word)

    # --------------------------------------------------------------
    # return ids of nodes in the tree (tokens in the sentece)
    def get_nodes(self):
        "return ids of nodes in the tree (tokens in the sentece)"

        return sorted(self.tree.nodes)[1:]

    # --------------------------------------------------------------
    # return number of nodes in the tree (tokens in the sentece), (plus one for the fake root)
    def get_n_nodes(self):
        "return number of nodes in the tree (tokens in the sentece), (plus one for the fake root)"

        return len(self.tree.nodes)

    # --------------------------------------------------------------
    # return the list of ancestors of a node
    def get_ancestors(self, n):
        "return the list of ancestors of a node"

        anc = []
        while n != 0:
            anc.append(n)
            n = self.tree.nodes[n]["head"]
        return anc

    # --------------------------------------------------------------
    # return the parent of a node
    def get_parent(self, n):
        "return the parent of a node"

        if n == 0:
            return None
        else:
            return self.tree.nodes[n]["head"]

    # --------------------------------------------------------------
    # return the children of a node
    def get_children(self, n):
        "return the children of a node"

        if self.tree is None:
            return []
        return [c for c in self.tree.nodes if self.get_parent(c) == n]

    # --------------------------------------------------------------
    # return the Lowest Common Subsumer of two nodes
    def get_LCS(self, n1, n2):
        "return the Lowest Common Subsumer of two nodes"

        # get ancestor list for each node
        a1 = self.get_ancestors(n1)
        a2 = self.get_ancestors(n2)
        # get first common element in both lists
        for i in range(len(a1)):
            for j in range(len(a2)):
                if a1[i] == a2[j]:
                    return a1[i]

        # (should never happen since tree root is always a common subsumer.)
        return None

    # --------------------------------------------------------------
    # get token heading the given sentence fragment (e.g. an entity span)
    def get_fragment_head(self, start, end):
        "get token heading the given sentence fragment (e.g. an entity span)"

        # find which tokens overlap the fragment
        overlap = set()
        for t in self.tree.nodes:
            tk_start, tk_end = self.get_offset_span(t)
            if tk_start <= start <= tk_end or tk_start <= end <= tk_end:
                overlap.add(t)

        head = None
        if len(overlap) > 0:
            # find head node among those overlapping the entity
            for t in overlap:
                if head is None:
                    head = t
                else:
                    head = self.get_LCS(head, t)

            # if found LCS does not overlap the entity, the parsing was wrong, forget it.
            if head not in overlap:
                head = None

        return head

    # --------------------------------------------------------------
    # get node word form
    def get_word(self, n):
        "get node word form"

        return (
            self.tree.nodes[n]["word"]
            if self.tree.nodes[n]["word"] is not None
            else "<none>"
        )

    # --------------------------------------------------------------
    # get node lemma
    def get_lemma(self, n):
        "get node lemma"

        return (
            self.tree.nodes[n]["lemma"]
            if self.tree.nodes[n]["lemma"] is not None
            else "<none>"
        )

    # --------------------------------------------------------------
    # get node syntactic function
    def get_rel(self, n):
        "get node syntactic function"

        return (
            self.tree.nodes[n]["rel"]
            if self.tree.nodes[n]["rel"] is not None
            else "<none>"
        )

    # --------------------------------------------------------------
    # get node PoS tag
    def get_tag(self, n):
        "get node PoS tag"

        return (
            self.tree.nodes[n]["tag"]
            if self.tree.nodes[n]["tag"] is not None
            else "<none>"
        )

    # --------------------------------------------------------------
    # get node offset
    def get_offset_span(self, n):
        "get node offset"

        if n == 0:
            return -1, -1
        else:
            return self.tree.nodes[n]["start"], self.tree.nodes[n]["end"]

    # --------------------------------------------------------------
    # check whether a token is a stopword
    def is_stopword(self, n):
        "check whether a token is a stopword"

        # if it is not a Noun, Verb, adJective, or adveRb, then it is a stopword
        return self.tree.nodes[n]["tag"][0] not in ["N", "V", "J", "R"]

    # --------------------------------------------------------------
    # check whether a token belongs to one of given entities
    def is_entity(self, n, entities):
        "check whether a token belongs to one of given entities"

        for e in entities:
            if (
                entities[e]["start"] <= self.tree.nodes[n]["start"]
                and self.tree.nodes[n]["end"] <= entities[e]["end"]
            ):
                return True
        return False

    # --------------------------------------------------------------
    # get span covered by a subtree rooted at node n
    def get_subtree_offset_span(self, n):
        "get span covered by a subtree rooted at node n"

        # if the node is a leaf, get its span
        left, right = self.get_offset_span(n)
        # if it is not a leaf, recurse into leftmost/rightmost children
        children = self.get_children(n)
        if children:
            l, r = self.get_subtree_offset_span(children[0])
            left = min(left, l)
            l, r = self.get_subtree_offset_span(children[-1])
            right = max(right, r)
        return left, right

    # --------------------------------------------------------------
    # get upwards path from n1 to n2 (returns list of node ids, upwards, excluding n2)
    def get_up_path(self, n1, n2):
        "get upwards path from n1 to n2"

        path = self.get_ancestors(n1)
        if n2 not in path:  # error, n2 is not ancestor of n1
            return None
        else:
            return path[: path.index(n2)]

    # --------------------------------------------------------------
    # get downwards path from n1 to n2 (return list of node ids, downwards, excluding n1)
    def get_down_path(self, n1, n2):
        "get downwards path from n1 to n2"

        path = self.get_up_path(n2, n1)
        if path is not None:  # if None, n1 was not ancestor of n2
            path.reverse()
        return path

    # --------------------------------------------------------------
    # print a readable version of the tree (useful for debugging and data exploration)
    def print(self, n=0, d=0, **kwargs):
        "print a readable version of the tree (useful for debugging and data exploration)"

        if n != 0:
            print(d * "   ", end="", **kwargs)
            print(
                self.get_rel(n) + "(" + self.get_lemma(n) + "_" + self.get_tag(n) + ")",
                **kwargs
            )
        for c in self.get_children(n):
            self.print(c, d + 1, **kwargs)
