# names: Bar Onn, Lee Griever

"""A class represnting a node in an AVL tree"""

class AVLNode(object):
	"""Constructor, you are allowed to add more fields. 

	@type value: str
	@param value: data of your node
	"""
	def __init__(self, value):
		self.value = value
		self.left = None
		self.right = None
		self.parent = None
		self.height = -1
		#added
		self.size = 0


	"""united function of returning a node's child, of side as requested in caller function

	@rtype: AVLNode
	@returns: None - if self is a virtual node, child otherwise
	"""	
	def getChild(self, child):
		if self.isRealNode():
			return child
		return None

	"""returns the left child

	@rtype: AVLNode
	@returns: the left child of self, None if there is no left child
	"""
	def getLeft(self):
		return self.getChild(self.left)

	"""returns the right child

	@rtype: AVLNode
	@returns: the right child of self, None if there is no right child
	"""
	def getRight(self):
		return self.getChild(self.right)

	"""returns the parent 

	@rtype: AVLNode
	@returns: the parent of self, None if there is no parent
	"""
	def getParent(self):
		return self.parent

	"""returns the value

	@rtype: str
	@returns: the value of self, None if the node is virtual
	"""
	def getValue(self):
		return self.value

	"""returns the height

	@rtype: int
	@returns: the height of self, -1 if the node is virtual
	"""
	def getHeight(self):
		return self.height

	#added
	"""returns the size of (how many nodes in) the subtree that self is it's root

	@rtype: int
	@returns: the size of self, 0 if the node is virtual
	"""
	def getSize(self):
		return self.size


	"""sets left child

	@type node: AVLNode
	@param node: a node
	"""
	def setLeft(self, node):
		if self.isRealNode():
			self.left = node

	"""sets right child

	@type node: AVLNode
	@param node: a node
	"""
	def setRight(self, node):
		if self.isRealNode():
			self.right = node

	"""sets parent

	@type node: AVLNode
	@param node: a node
	"""
	def setParent(self, node):
		self.parent = node

	"""sets value

	@type value: str
	@param value: data
	"""
	def setValue(self, value):
		self.value = value

	"""sets the height of the node

	@type h: int
	@param h: the height
	"""
	def setHeight(self, h):
		self.height = h

	"""sets the size of the node

	@type size: int
	@param size: the size
	"""
	def setSize(self, size):
		self.size = size
	
	"""returns whether self is not a virtual node 

	@rtype: bool
	@returns: False if self is None or a virtual node, True otherwise.
	"""
	def isRealNode(self):
		return self != None and self.height > -1
	
	"""creates and returns new non-virtual node for insertion purposes, that will function as leaf
	"""
	def createRealNode(val):
		new_node = AVLNode(val)
		new_node.setHeight(0)
		new_node.setSize(1)
		virtual_node_left = AVLNode(None)
		virtual_node_right = AVLNode(None)
		new_node.setLeft(virtual_node_left)
		new_node.setRight(virtual_node_right)
		virtual_node_left.setParent(new_node)
		virtual_node_right.setParent(new_node)
		return new_node



"""
A class implementing the ADT list, using an AVL tree.
"""

class AVLTreeList(object):

	"""
	Constructor, you are allowed to add more fields.  

	"""
	def __init__(self):
		self.root = None
		# added
		self.min = None
		self.max = None

	"""returns whether the list is empty

	@rtype: bool
	@returns: True if the list is empty, False otherwise
	"""
	def empty(self):
		return (self.root == None) or (not self.root.isRealNode())

	"""Recursive implementation of Tree-Select function

	@rtype: AVL node
	@returns: the k'th ranked node in the subtree which node is it's root
	@complexity: O(log(n))
	"""
	def treeSelectRec(self, node, k):
		if not AVLNode.isRealNode(node.getLeft()):
			r = 1
		else:
			r = node.getLeft().getSize() + 1
		if k == r:
			return node
		elif k < r:
			return self.treeSelectRec(node.getLeft(), k)
		else:
			return self.treeSelectRec(node.getRight(), k - r)

	"""retrieves the value of the i'th item in the list

	@type i: int
	@pre: 0 <= i < self.length()
	@param i: index in the list
	@rtype: str
	@returns: the value of the i'th item in the list
	@complexity: O(log(n))
	"""
	def retrieve(self, i):
		if i >= self.length() or i < 0:
			return None
		node = self.treeSelectRec(self.root, i + 1)
		if AVLNode.isRealNode(node):
			return node.getValue()
		return None

	"""retrieves the node in index i in the list

	@type i: int
	@pre: 0 <= i < self.length()
	@param i: index in the list
	@rtype: AVLNode
	@returns: the node of the i'th item in the list
	@complexity: O(log(n))
	"""
	def retrieveNodeInIndex(self, i):
		if i >= self.length():
			return None
		node = self.treeSelectRec(self.root, i + 1)
		if AVLNode.isRealNode(node):
			return node
		return None
	
	"""returns min node in the subtree that node is it's root
	@rtype: AVL Node
	@complexity: O(log(n))
	"""
	def minNode(self, node):
		if not AVLNode.isRealNode(node):
			return None
		while AVLNode.isRealNode(node.getLeft()):
			node = node.getLeft()
		return node
	
	"""returns max node in the subtree that node is it's root
	@rtype: AVL Node
	@complexity: O(log(n))
	"""
	def maxNode(self, node):
		if not AVLNode.isRealNode(node):
			return None
		while AVLNode.isRealNode(node.getRight()):
			node = node.getRight()
		return node
	
	"""returns predecessor of node
	@rtype: AVL Node
	@complexity: O(log(n))
	"""
	def predecessor(self, node):
		if AVLNode.isRealNode(node.getLeft()):
			return self.maxNode(node.getLeft())
		
		parent = node.getParent()
		while AVLNode.isRealNode(parent) and node == parent.getLeft():
			node = parent
			parent = node.getParent()

		return parent

	"""returns successor of node
	@rtype: AVL Node
	@complexity: O(log(n))
	"""
	def successor(self, node):
		if AVLNode.isRealNode(node.getRight()):
			return self.minNode(node.getRight())
		
		parent = node.getParent()
		while AVLNode.isRealNode(parent) and node == parent.getRight():
			node = parent
			parent = node.getParent()
		
		return parent

	"""returns true if node is the right child of it's parent, False otherwise.
	if parent is None returns default value: False.
	"""
	def isRightChild(self, node):
		if node.getParent() == None:
			# default value
			return False
		return node.getParent().getRight() == node

	"""sets parent's left/right child as child, and child's parent as parent
	"""
	def setParentChild(self, parent, child, isRightChild):
		# if parent == child or child is al no changes needed
		if (parent == child):
			return
		# in case child is root, child's parent is set to None
		if parent != None:
			# if child was root, then update root to parent
			if child == self.getRoot():
				self.setRoot(parent)
			if isRightChild:
				# orphanize parent's original child
				parent.getRight().setParent(None)
				# set parent new child
				parent.setRight(child)
			else:
				parent.getLeft().setParent(None)
				parent.setLeft(child)
		else:
			self.setRoot(child)
		child_original_parent = child.getParent()
		# kiling original parent child
		if (child_original_parent != None) and (child_original_parent != parent):
			if self.isRightChild(child):
				child_original_parent.setRight(AVLNode(None))
			else:
				child_original_parent.setLeft(AVLNode(None))
		child.setParent(parent)


	"""detaches parent and child, sets virtual node as parent's new child, and sets None as child's new parent.
	@pre: child.getParent() == parent
	"""
	def detachParentChild(self, parent, child):
		if parent == None:
			return
		isRightChild = self.isRightChild(child)
		child.setParent(None)
		if isRightChild:
			parent.setRight(AVLNode(None))
		else:
			parent.setLeft(AVLNode(None))

	"""calculates Balance Factor of node
	"""
	def calcBF(self, node):
		return node.getLeft().getHeight() - node.getRight().getHeight()

	"""updates size and height attributes of every ancestor of node until the root
	@complexity: O(log(n))
	"""
	def updateSizeHeightUpwards(self, node):
		num_of_changed_height_nodes = 0
		while node != None:
			node.setSize(1 + node.getLeft().getSize() + node.getRight().getSize())
			new_height = 1 + max(node.getLeft().getHeight(), node.getRight().getHeight())
			if new_height != node.getHeight():
				num_of_changed_height_nodes += 1
			node.setHeight(new_height)

			node = node.getParent()
		return num_of_changed_height_nodes


	"""updates size and height attributes of given node
	adds node to "changed_height_nodes" if node changed it's height, False otherwise
	@param changed_height_nodes: Set
	"""
	def updateSizeHeight(self, node, changed_height_nodes):
		if AVLNode.isRealNode(node):
			# update size
			node.setSize(1 + node.getLeft().getSize() + node.getRight().getSize())
			# update height
			old_height = node.getHeight()
			new_height = 1 + max(node.getLeft().getHeight(), node.getRight().getHeight())
			if new_height != old_height:
				node.setHeight(new_height)
				changed_height_nodes.add(node)


	"""performs right rotation if isRightRotation == True, else performs left rotation
	"""
	def rotate(self, B, A, isRightRotation):
		isRightChild = self.isRightChild(B)

		if isRightRotation:
			self.setParentChild(B, A.getRight(), False)
			A.setRight(B)
		else:
			self.setParentChild(B, A.getLeft(), True)
			A.setLeft(B)

		if isRightChild:
			self.setParentChild(B.getParent(), A, True)
		else:
			self.setParentChild(B.getParent(), A, False)
		B.setParent(A)

		# update height and size of A,B
		B.setSize(1 + B.getLeft().getSize() + B.getRight().getSize())
		B.setHeight(1 + max(B.getLeft().getHeight(), B.getRight().getHeight()))
		A.setSize(1 + A.getLeft().getSize() + A.getRight().getSize())
		A.setHeight(1 + max(A.getLeft().getHeight(), A.getRight().getHeight()))



	"""Diagnoses what kind of rotation is needed and calles "rotate" method to perform it
	counts and returns rotation actions
	"""
	def rotationDiagnoseAndFix(self, criminal, bf):
		# found an AVL criminal, so a single rotation must be performed
		counter = 1

		if bf == 2:
			child_bf = self.calcBF(criminal.getLeft())
			if child_bf == 1 or child_bf == 0:
				# right rotation
				self.rotate(criminal, criminal.getLeft(), True)
			elif child_bf == -1:
				# left then right rotation
				self.rotate(criminal.getLeft(), criminal.getLeft().getRight(), False)
				self.rotate(criminal, criminal.getLeft(), True)
				counter += 1

		elif bf == -2:
			child_bf = self.calcBF(criminal.getRight())
			if child_bf == -1 or child_bf == 0:
				# left rotation
				self.rotate(criminal, criminal.getRight(), False)
			elif child_bf == 1:
				# right then left rotation
				self.rotate(criminal.getRight(), criminal.getRight().getLeft(), True)
				self.rotate(criminal, criminal.getRight(), False)
				counter += 1
			
		return counter


	"""rebalances the tree after insertion/deletion back into an AVL tree.
	returns the number of balance operations performed
	@complexity: O(log(n))
	"""
	def rebalance(self, curr_node, isInsert):
		op_count = 0
		changed_height_nodes = set()
		while curr_node != None:
			self.updateSizeHeight(curr_node, changed_height_nodes)
			bf = self.calcBF(curr_node)
			if abs(bf) < 2:
				if curr_node not in changed_height_nodes:
					# legal BF and height did not change - no rotations needed from this node upwards
					break
				else:
					op_count += 1
					curr_node = curr_node.getParent()
			else:
				# found an AVL criminal!
				op_count += self.rotationDiagnoseAndFix(curr_node, bf)
				if isInsert:
					# in insert, only one rotation is needed for rebalance
					break
				# passing over A
				if curr_node.getParent() == None:
					break
				curr_node = curr_node.getParent().getParent()
					
		op_count += self.updateSizeHeightUpwards(curr_node)
		
		return op_count
		

	"""inserts val at position i in the list

	@type i: int
	@pre: 0 <= i <= self.length()
	@param i: The intended index in the list to which we insert val
	@type val: str
	@param val: the value we inserts
	@rtype: int
	@returns: the number of rebalancing operation due to AVL rebalancing
	@complexity: O(log(n))
	"""
	def insert(self, i, val):
		inserted_node = AVLNode.createRealNode(val)
		if i == 0:
			self.min = inserted_node
		if i == self.length():
			self.max = inserted_node

		if self.empty():
			self.setRoot(inserted_node)
			return 0

		elif i == self.length():
			curr_last = self.maxNode(self.root)
			self.setParentChild(curr_last, inserted_node, True)
		else:
			curr_node = self.retrieveNodeInIndex(i)
			if not AVLNode.isRealNode(curr_node.getLeft()):
				self.setParentChild(curr_node, inserted_node, False)
			else:
				pred = self.predecessor(curr_node)
				self.setParentChild(pred, inserted_node, True)
			
		# caliing rebalance, isInsert = true
		startOfRebalanceNode = inserted_node.getParent()
		op_count = self.rebalance(startOfRebalanceNode, True)
		return op_count
		
	"""returns True is node is leaf, False otherwise
	"""
	def isLeaf(self, node):
		return not (AVLNode.isRealNode(node.getLeft()) or AVLNode.isRealNode(node.getRight()))
	
	
	"""deletes the i'th item in the list

	@type i: int
	@pre: 0 <= i < self.length()
	@param i: The intended index in the list to be deleted
	@rtype: int
	@returns: the number of rebalancing operation due to AVL rebalancing
	@complexity: O(log(n))
	"""
	def delete(self, i):
		if i < 0 or i >= self.length():
			return -1
		
		deleted_node = self.retrieveNodeInIndex(i)
		deleted_height = deleted_node.getHeight()
		isRightChild = self.isRightChild(deleted_node)
		startOfRebalanceNode = deleted_node.getParent()
		succesor = None
		case3 = False
		# case1 - node is a leaf
		if self.isLeaf(deleted_node):
			self.setParentChild(deleted_node.getParent(), AVLNode(None), isRightChild)
		# case2 - node has 1 child
		# op "!=" functions as xor for boolean
		elif AVLNode.isRealNode(deleted_node.getLeft()) != AVLNode.isRealNode(deleted_node.getRight()):
			if AVLNode.isRealNode(deleted_node.getLeft()):
				self.setParentChild(deleted_node.getParent(), deleted_node.getLeft(), isRightChild)
			else:
				self.setParentChild(deleted_node.getParent(), deleted_node.getRight(), isRightChild)
		# case3 - node has 2 children
		else:
			case3 = True
			succesor = self.successor(deleted_node)
			succesorParent = succesor.getParent()
			successorRightChild = succesor.getRight()
			isSuccesorRightChild = self.isRightChild(succesor)
			deletedNodeParent = deleted_node.getParent()

			self.setParentChild(succesor, deleted_node.getLeft(), False)
			self.setParentChild(succesor, deleted_node.getRight(), True)
			self.setParentChild(deletedNodeParent, succesor, isRightChild)
			# handling edge case: deletedNode is the direct parent of his successor
			if succesorParent == deleted_node:
				succesorNewHeight = 1 + max(succesor.getRight().getHeight(), succesor.getLeft().getHeight())
				SuccessorChanged = succesor.getHeight() != succesorNewHeight
				if SuccessorChanged:
					startOfRebalanceNode = succesor
				else:
					self.updateSizeHeight(succesor, set())
					startOfRebalanceNode = succesor.getParent()
			else:
				self.setParentChild(succesorParent, successorRightChild, isSuccesorRightChild)
				startOfRebalanceNode = succesorParent

		# caling rebalnce, isInsert = false
		op_count = self.rebalance(startOfRebalanceNode, False)
		if case3 and (deleted_height == succesor.getHeight()):
			op_count -= 1
		self.min = self.minNode(self.getRoot())
		self.max = self.maxNode(self.getRoot())
		return op_count

		

	"""returns the value of the first item in the list

	@rtype: str
	@returns: the value of the first item, None if the list is empty
	"""
	def first(self):
		if self.empty():
			return None
		return self.min.getValue()

	"""returns the value of the last item in the list

	@rtype: str
	@returns: the value of the last item, None if the list is empty
	"""
	def last(self):
		if self.empty():
			return None
		return self.max.getValue()

	"""performs in-order walk in the tree, creating a sorted array of it's values
	@complexity: O(n)
	"""
	def inOrder(self, node, array):
		if AVLNode.isRealNode(node):
			self.inOrder(node.getLeft(), array)
			array.append(node.getValue())
			self.inOrder(node.getRight(), array)

		return array

	"""returns an array representing list 

	@rtype: list
	@returns: a list of strings representing the data structure
	@complexity: O(n)
	"""
	def listToArray(self):
		array = []
		sortedArray = self.inOrder(self.root, array)
		return sortedArray

	"""returns the length of the list

	@rtype: int
	@returns: the size of the list
	"""
	def length(self):
		if self.empty():
			return 0
		return self.root.getSize()

	"""joining T1, T2 into AVLTreeList, in the following manner: "T1 + x + T2"
	@type T1: AVLTreeList
	@type T2: AVLTreeList
	@type x: AVLNode
	@complexity: O(log(n)) (tighter bound: O(height(T1)-height(T2) + 1))
	"""
	def joinAVL(self, x, T2):
		T1 = self
		joinedTree = AVLTreeList()
		if T1.empty():
			T1_root = AVLNode(None) # virtual_node
			T1_min = x
		else:
			T1_root = T1.getRoot()
			T1_min = T1.min
		if T2.empty():
			T2_root = AVLNode(None) # virtual_node
			T2_max = x
		else:
			T2_root = T2.getRoot()
			T2_max = T2.max
		
		h1 = T1_root.getHeight()
		h2 = T2_root.getHeight()
		
		# naive joining, AVL BF condition not violated
		if abs(h1 - h2) <= 1:
			self.setParentChild(x, T1_root, False)
			self.setParentChild(x, T2_root, True)
			# was to set root's parent to none, but handeled directtly in concat and split
			root = x

		else:
			# on start: a := shorter tree's root, b := higher tree's root
			if h1 < h2: 
				h_min = h1
				b = T2_root
				a = T1_root
				getCorrectChild = lambda node: node.getLeft()
				isT1Smaller = True
				root = T2.getRoot()
			else:
				h_min = h2
				b = T1_root
				a = T2_root
				getCorrectChild = lambda node: node.getRight()
				isT1Smaller = False
				root = T1.getRoot()
					
			while b.getHeight() > h_min:
				b = getCorrectChild(b)

			# arrived to b := the first node in height <= height of smaller tree
			bParent = b.getParent()
			joinedTree.setParentChild(x, a, not isT1Smaller)
			joinedTree.setParentChild(x, b, isT1Smaller)
			joinedTree.setParentChild(bParent, x, not isT1Smaller)

		joinedTree.setRoot(root)
		joinedTree.updateSizeHeight(x, set())
		joinedTree.rebalance(x.getParent(), False)

		# set min, max attributes of joined tree
		
		joinedTree.min = T1_min
		joinedTree.max = T2_max
				
		return joinedTree



	"""splits the list at the i'th index

	@type i: int
	@pre: 0 <= i < self.length()
	@param i: The intended index in the list according to whom we split
	@rtype: list
	@returns: a list [left, val, right], where left is an AVLTreeList representing the list until index i-1,
	right is an AVLTreeList representing the list from index i+1, and val is the value at the i'th index.
	@complexity: O(log(n))
	"""
	def split(self, i, getStat = False):
		x = self.retrieveNodeInIndex(i)
		val = x.getValue()

		smaller = AVLTreeList()
		smaller.setRoot(x.getLeft())
		bigger = AVLTreeList()
		bigger.setRoot(x.getRight())
		currNode = x
		self.detachParentChild(x, x.getRight())
		self.detachParentChild(x, x.getLeft())
		parent = currNode.getParent()
		isRightChild = self.isRightChild(currNode)

		while parent != None:
			grandparent = parent.getParent()
			isParentRightChild = self.isRightChild(parent)
			self.detachParentChild(grandparent, parent)
			addedSubTree = AVLTreeList()
			# yellow tree in class example
			if isRightChild:
				addedSubTree.setRoot(parent.getLeft())
				self.detachParentChild(parent, parent.getLeft())
				self.detachParentChild(parent, parent.getRight())
				smaller = addedSubTree.joinAVL(parent, smaller)
			# pink tree in class example
			else:
				addedSubTree.setRoot(parent.getRight())
				self.detachParentChild(parent, parent.getLeft())
				self.detachParentChild(parent, parent.getRight())
				bigger = bigger.joinAVL(parent, addedSubTree)
			currNode = parent
			parent = grandparent
			isRightChild = isParentRightChild

		# fix root parent to none
		smaller.setParentChild(None, smaller.getRoot(), False)
		bigger.setParentChild(None, bigger.getRoot(), False)

		bigger.min = bigger.minNode(bigger.getRoot())
		bigger.max = bigger.maxNode(bigger.getRoot())
		smaller.min = smaller.minNode(smaller.getRoot())
		smaller.max = smaller.maxNode(smaller.getRoot())

		return [smaller, val, bigger]

	"""concatenates lst to self

	@type lst: AVLTreeList
	@param lst: a list to be concatenated after self
	@rtype: int
	@returns: the absolute value of the difference between the height of the AVL trees joined
	@complexity: O(log(n))
	"""
	def concat(self, lst):
		if self.getRoot() == None:
			h1 = -1
		else:
			h1 = self.getRoot().getHeight()
		if lst.getRoot() == None:
			h2 = -1
		else:
			h2 = lst.getRoot().getHeight()
		res = abs(h1 - h2)
		# self is empty list, convert it to be lst
		if self.empty():
			self.setRoot(lst.getRoot())
		# if lst is empty, do nothing
		elif not lst.empty():
			# x gets the value of last node
			last_index = self.length() - 1
			x = self.retrieveNodeInIndex(last_index)
			# x = self.maxNode(self.getRoot())
			self.delete(last_index)
			joinedTree = self.joinAVL(x, lst)
			self.setRoot(joinedTree.getRoot())
			self.setParentChild(None, self.getRoot(), False)
		
		self.min = self.minNode(self.getRoot())
		self.max = self.maxNode(self.getRoot())
		return res

	"""searches for a *value* in the list

	@type val: str
	@param val: a value to be searched
	@rtype: int
	@returns: the first index that contains val, -1 if not found.
	@complexity: O(n)
	"""
	def search(self, val):
		sorted_array = self.listToArray()
		for i,value in enumerate(sorted_array):
			if val == value:
				return i
		return -1


	"""returns the root of the tree representing the list

	@rtype: AVLNode
	@returns: the root, None if the list is empty
	"""
	def getRoot(self):
		if self.empty():
			return None
		return self.root


	"""sets the root of the tree representing the list to be root
	@type root: AVLNode
	"""
	def setRoot(self, root):
		if AVLNode.isRealNode(root):
			self.root = root
		else:
			self.root = None