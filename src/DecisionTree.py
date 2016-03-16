import math
import numpy as np
import copy

class DTreeNode:
	def __init__(self, feature_data, target_data, col_len=-1):
		self.feature_data = feature_data
		self.target_data = target_data
		self.children = {} #List of Node Children
		self.output_dict = None
		self.split_on = -1
		self.label = None
		self.col_len = col_len

class DecisionTree:
	def __init__(self, debug=False, validation=False):
		self.num_decisions = 0
		self.head_node = None
		self.debug = debug
		self.validation = validation
		self.validation_accuracy = 0
	def train(self, instances):
		targets = instances[:, -1]
		inputs = instances[:, :-1]

		self.length = len(targets)
		self.num_decisions = len(set(targets))

		if self.validation:
			percent_for_training = .75
		
			training_instances = instances[0:percent_for_training * len(instances)]
			validation_instances = instances[percent_for_training * len(instances):-1]

			targets = training_instances[:, -1]
			inputs = training_instances[:, :-1]

		self.head_node = DTreeNode(inputs, targets, col_len=len(inputs[0]))
		self.create_decision_tree(self.head_node)
		depth, nodes = self.get_tree_depth()
		print "Un-Pruned Tree Depth: ",  depth, " Nodes: ", nodes


		if self.validation:
			self.validation_accuracy =  self.validate_data(validation_instances)
			self.prune_tree(validation_instances)
		if self.debug:
			self.print_tree()

	def create_decision_tree(self, head_node):
		self.find_children(head_node, 100)

	def most_common(self, lst):
		return max(set(lst), key=lst.count)

	def find_children(self, curr_node, max_depth=0):
		if len(set(curr_node.target_data)) == 1:
			curr_node.label = curr_node.target_data[0]
			return "Finished with perfect leaf"
		elif curr_node.col_len == 0:
			if isinstance(curr_node.target_data, list):
				curr_node.label = self.most_common(curr_node.target_data)
			else:
				curr_node.label = self.most_common(np.ndarray.tolist(curr_node.target_data))
			return "Finished out of necessity"

		#Forcing a max depth for experimentation
		if self.head_node.col_len - curr_node.col_len == max_depth:
			if isinstance(curr_node.target_data, list):
				curr_node.label = self.most_common(curr_node.target_data)
			else:
				curr_node.label = self.most_common(np.ndarray.tolist(curr_node.target_data))
			return "Forced the Max Depth"

		output_entropy = self.get_output_entropy(curr_node)
		info_gain_arr = []
		col_dicts = []
		col_len = len(curr_node.feature_data[0])
		for col_index in xrange(col_len):
			col =  np.array(curr_node.feature_data)[:,col_index]
			col_dict = self.create_column_dict(col)
			col_dicts.append( col_dict )
			info_gain = self.get_info_gain(col_dict, curr_node)
			info_gain_arr.append(output_entropy - info_gain)
		curr_node.split_on = np.argmax(info_gain_arr)

		for key, x in col_dicts[curr_node.split_on].items():
			if not isinstance(x, int): #not the output_dict['total']
				child_inputs = [curr_node.feature_data[y] for y in x['indeces']]
				child_targets = [curr_node.target_data[y] for y in x['indeces']]
				child_node = DTreeNode(child_inputs, child_targets, col_len= curr_node.col_len - 1 )
				curr_node.children[key] = child_node
				self.find_children( child_node, max_depth )


	def get_output_entropy(self, head_node):
		self.output_dict = self.create_column_dict(head_node.target_data)
		return self.get_single_entropy(self.output_dict)

	def get_info_gain(self, col_dict, head_node):
		info_gain = 0
		for row_key, row_value in col_dict.items():
			#print "Attribute ", row_key
			if not isinstance(row_value, int): #not the output_dict['total']
				converted = self.create_column_dict( [head_node.target_data[x] for x in row_value['indeces']])
				#print "Entropy: ", self.get_single_entropy(converted)
				info_gain += (row_value['total'] / float(self.output_dict['total'])) * self.get_single_entropy(converted)
		return info_gain

	def create_column_dict(self, column_data):
		col_dict = {}
		col_dict["total"] = len(column_data)
		index = 0
		for data in column_data:
			if math.isnan(data):
				data = "unknown"
			if not col_dict.get(data):
				col_dict[data] = {'indeces' : [], 'total': 0}
			col_dict[data]['indeces'].append(index)
			col_dict[data]['total'] += 1
			index += 1
		return col_dict

	def validate_data(self, validation_set):
		num_correct = 0.0
		for instance in validation_set:
			features = instance[:-1]
			goal = instance[-1]

			prediction = self.predict(features)
			num_correct += 1 if prediction == goal else 0
		accuracy = (num_correct / len(validation_set)) * 100
		return accuracy

	def get_single_entropy(self, output_dict):
		total = 0
		for data in output_dict.values():
			if not isinstance(data, int): #not the output_dict['total']
				x = float(data["total"]) / output_dict["total"]
				if x == 0:
					pass
				else:
					total+= -x * math.log(x, 2)
		return total

	def search_tree(self, curr_node, inputs):
		if curr_node.label == None:
			key = inputs[curr_node.split_on]
			if math.isnan(key):
				if not curr_node.children.get('unknown'):
					return curr_node.label or 1
				next_node = curr_node.children["unknown"]
			else:
				if not curr_node.children.get(key):
					return curr_node.label or 1
				next_node = curr_node.children[key]
			prediction = self.search_tree(next_node, inputs)
		else:
			return curr_node.label
		return prediction

	def prune_tree(self, validation_set):
		self.prune_node(self.head_node, validation_set)
		depth, nodes = self.get_tree_depth()
		print "Pruned Tree Depth: ", depth, " Number of nodes: ", nodes

	def prune_node(self, curr_node, validation_set):

		if curr_node != self.head_node:
			if curr_node.split_on != -1:
				temp_node = copy.deepcopy(curr_node)
				curr_node.split_on = -1
				curr_node.label = self.most_common(curr_node.target_data)
				curr_node.children = {}
				#test on validation
				new_acc = self.validate_data(validation_set)
				#print "New Accuracy: ", new_acc
				#print "Old Accuracy: ", self.validation_accuracy
				if new_acc < self.validation_accuracy:
					#if validation is worse, return it 
					curr_node = temp_node

		for key, child in curr_node.children.items():
			curr_node.children[key] = self.prune_node(child, validation_set)
		
		return curr_node

	def get_tree_depth(self):
		return self.get_node_depth(self.head_node)

	def get_node_depth(self, curr_node, orig_depth=1, nodes=1):
		if curr_node.label != None:
			return orig_depth, nodes
		depths = []
		for child in curr_node.children.values():
			depth, nodes = self.get_node_depth(child, orig_depth+1, nodes+1)
			depths.append( depth )
		return max(depths), nodes


	def print_tree(self):
		print "\n\n\n ------------------------------------------- \n\n\n"
		self.print_node(self.head_node)
		print "\n\n\n ------------------------------------------- \n\n\n"

	def print_node(self, curr_node, tabs=0):
		if curr_node.split_on != -1:
			print "\t"* tabs, "Node split on attribute: ", curr_node.split_on
		else:
			print "\t"* tabs, "Node with Label: ", curr_node.label, "\n"

		if curr_node.children:
			print "\t"* tabs, "Children include: \n"
		for child in curr_node.children.values():
			self.print_node(child, tabs+1)

	def predict(self, inputs):
		return self.search_tree(self.head_node, inputs)