import xml.etree.ElementTree as ET
import re

def get_focused_package_text(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for node in root.iter("node"):
        if node.attrib.get("focused") == "true":
            # class_name = node.attrib.get("class")
            text_name = node.attrib.get("text")
            package_name = node.attrib.get("package")
            return package_name, text_name
        
    return None, None

def parse_xml_hierarchy(xml_string):
    # Define the namespace pattern to remove
    namespace_pattern = re.compile(r'\s+xmlns="[^"]+"')
    # Remove namespaces for easier parsing
    xml_string = namespace_pattern.sub('', xml_string)
    
    # Parse the XML
    root = ET.fromstring(xml_string)
    
    # Initialize lists to store different types of UI elements
    clickable_elements = []
    long_clickable_elements = []
    scrollable_elements = []
    editable_elements = []
    
    # Function to recursively collect descriptive information from all descendants
    def collect_descriptions_from_descendants(element):
        # Separate collections for different types of descriptions
        descriptions = []
        text_descriptions = []
        content_desc_descriptions = []
        resource_id_descriptions = []
        
        # Check this element's descriptive properties
        element_text = element.get('text', '')
        element_desc = element.get('content-desc', '')
        element_resource_id = element.get('resource-id', '')
        
        # Add non-empty values to their respective collections
        if element_text:
            text_descriptions.append(element_text)
        if element_desc:
            content_desc_descriptions.append(element_desc)
        if element_resource_id and not element_resource_id.endswith(':id/'):
            id_parts = element_resource_id.split('/')
            clean_id = id_parts[-1] if len(id_parts) > 1 else element_resource_id
            resource_id_descriptions.append(clean_id)
        
        # Recursively check all children
        for child in element:
            child_text, child_desc, child_res_id = collect_descriptions_from_descendants(child)
            text_descriptions.extend(child_text)
            content_desc_descriptions.extend(child_desc)
            resource_id_descriptions.extend(child_res_id)
        
        return text_descriptions, content_desc_descriptions, resource_id_descriptions
    
    # Recursively traverse the XML tree
    def traverse_element(element, path=""):
        # Get element details
        element_class = element.get('class', '')
        element_text = element.get('text', '')
        element_desc = element.get('content-desc', '')
        element_resource_id = element.get('resource-id', '')
        element_bounds = element.get('bounds', '')
        
        # Get a clean resource ID for display
        clean_resource_id = ""
        if element_resource_id and not element_resource_id.endswith(':id/'):
            id_parts = element_resource_id.split('/')
            clean_resource_id = id_parts[-1] if len(id_parts) > 1 else element_resource_id
        
        # Check if this element has descriptive information
        has_description = bool(element_text or element_desc or clean_resource_id)
        
        # If no descriptive information, collect from all descendants
        if True:#not has_description:
            descendant_text_list, descendant_desc_list, descendant_id_list = collect_descriptions_from_descendants(element)
            
            # Combine into one prioritized list
            # all_descriptions = text_descriptions + desc_descriptions + id_descriptions
            
            # Remove duplicates while preserving order
            # seen = set()
            # unique_descriptions = []
            # for desc in all_descriptions:
            #     if desc not in seen:
            #         seen.add(desc)
            #         unique_descriptions.append(desc)
            
            # descendant_text = " - ".join(unique_descriptions)
        # Use element's own description first, fall back to descendants' if needed
        # display_text = element_text or element_desc or clean_resource_id or descendant_text
                # First use ALL text (element and its children)
        all_text = []
        # if element_text:
        #     all_text.append(element_text)
        all_text.extend(descendant_text_list)
        
        # If no text is available, fall back to descriptions
        all_desc = []
        # if not all_text:
        #     if element_desc:
        #         all_desc.append(element_desc)
        all_desc.extend(descendant_desc_list)
        
        # If no descriptions either, fall back to resource IDs
        all_resource_ids = []
        # if not all_text and not all_desc:
        #     if clean_resource_id:
        #         all_resource_ids.append(clean_resource_id)
        all_resource_ids.extend(descendant_id_list)
        
        # Create display text from the appropriate list
        display_text = ""
        if all_text:
            display_text = " - ".join(all_text)
        elif all_desc:
            display_text = " - ".join(all_desc)
        elif all_resource_ids:
            display_text = " - ".join(all_resource_ids)
        display_text = f"\"{display_text}\""
        # Create a concise representation of the element
        element_repr = {
            'class': element_class,
            'text': element_text,
            'desc': element_desc,
            'resource_id': element_resource_id,
            'display_text': display_text,
            'bounds': element_bounds,
            'path': path
        }
        
        # Check for clickable elements
        if element.get('clickable') == 'true':
            clickable_elements.append(element_repr)
        
        # Check for long-clickable elements
        if element.get('long-clickable') == 'true':
            long_clickable_elements.append(element_repr)
        
        # Check for scrollable elements
        if element.get('scrollable') == 'true':
            scrollable_elements.append(element_repr)
        
        # Check for potentially editable elements
        # if ('EditText' in element_class or 
        #     (element.get('focusable') == 'true' and 
        #      ('edit' in element_resource_id.lower() or 
        #       'input' in element_resource_id.lower() or
        #       'text' in element_resource_id.lower()))):
        #     editable_elements.append(element_repr)
        if False:
            editable_elements.append(element_repr)
        
        # Recursively process child elements
        for i, child in enumerate(element):
            child_path = f"{path}/{i}" if path else str(i)
            traverse_element(child, child_path)
    
    # Start traversal from the root
    traverse_element(root)
    
    return {
        'clickable': clickable_elements,
        'long_clickable': long_clickable_elements,
        'scrollable': scrollable_elements,
        'editable': editable_elements
    }

def display_ui_elements(elements_dict):
    """
    Display the extracted UI elements in a readable format with unified ID numbering for actions
    Returns a string containing all the output instead of printing directly
    Numbers are assigned to each action rather than each element
    """
    import re
    output = ""
    current_id = 1
    
    
    if len(elements_dict['clickable']) != 0:
        output += "=== CLICKABLE ELEMENTS ===\n"
        for elem in elements_dict['clickable']:
            # output += f"{elem['text'] or elem['desc'] or elem['resource_id']} ({elem['class']})\n"
            # output += f"{elem['display_text']} ({elem['class']})\n"
            output += f"{elem['display_text']} ({elem['class']})\n"
            output += f"Bounds: {elem['bounds']}\n"
            # Extract and calculate the middle point for the action
            bounds = elem['bounds']
            # Extract coordinates using regex
            coordinates = re.findall(r'\d+', bounds)
            if len(coordinates) == 4:
                x1, y1, x2, y2 = map(int, coordinates)
                middle_x = (x1 + x2) // 2
                middle_y = (y1 + y2) // 2
                output += f"{current_id}. Action: click,{middle_x},{middle_y}\n"
                current_id += 1
            
            output += "\n"
    
    # if len(elements_dict['long_clickable']) != 0:
    #     output += "=== LONG-CLICKABLE ELEMENTS ===\n"
    #     for elem in elements_dict['long_clickable']:
    #         # output += f"{elem['text'] or elem['desc'] or elem['resource_id']} ({elem['class']})\n"
    #         # output += f"{elem['display_text']} ({elem['class']})\n"
    #         output += f"{elem['display_text']} ({elem['class']})\n"
    #         output += f"Bounds: {elem['bounds']}\n"
    #         # Extract and calculate the middle point for the action
    #         bounds = elem['bounds']
    #         # Extract coordinates using regex
    #         coordinates = re.findall(r'\d+', bounds)
    #         if len(coordinates) == 4:
    #             x1, y1, x2, y2 = map(int, coordinates)
    #             middle_x = (x1 + x2) // 2
    #             middle_y = (y1 + y2) // 2
    #             output += f"{current_id}. Action: swipe, {middle_x}, {middle_y}, {middle_x}, {middle_y}\n"
    #             current_id += 1

    #         output += "\n"
    
    # if len(elements_dict['scrollable']) != 0:
    #     output += "=== SCROLLABLE ELEMENTS ===\n"
    #     for elem in elements_dict['scrollable']:
    #         # output += f"{elem['text'] or elem['desc'] or elem['resource_id']} ({elem['class']})\n"
    #         # output += f"{elem['display_text']} ({elem['class']})\n"
    #         output += f"{elem['display_text']} ({elem['class']})\n"
    #         output += f"Bounds: {elem['bounds']}\n"
    #         element_class = elem['class']
    #         bounds = elem['bounds']
    #         coordinates = re.findall(r'\d+', bounds)
            
    #         if len(coordinates) == 4:
    #             x1, y1, x2, y2 = map(int, coordinates)
    #             middle_y = (y1 + y2) // 2
    #             middle_x = (x1 + x2) // 2
                
    #             # Check if it's HorizontalScrollView
    #             if 'HorizontalScrollView' in element_class:
    #                 left_x = x1 + 10
    #                 right_x = x2 - 10
    #                 output += f"{current_id}. Action from left to right: swipe, {left_x}, {middle_y}, {right_x}, {middle_y}\n"
    #                 current_id += 1
    #                 output += f"{current_id}. Action from right to left: swipe, {right_x}, {middle_y}, {left_x}, {middle_y}\n"
    #                 current_id += 1
                
    #             # Check if it's VerticalScrollView
    #             elif 'VerticalScrollView' in element_class or 'ScrollView' in element_class:
    #                 top_y = y1 + 10
    #                 bottom_y = y2 - 10
    #                 output += f"{current_id}. Action from top to bottom: swipe, {middle_x}, {top_y}, {middle_x}, {bottom_y}\n"
    #                 current_id += 1
    #                 output += f"{current_id}. Action from bottom to top: swipe, {middle_x}, {bottom_y}, {middle_x}, {top_y}\n"
    #                 current_id += 1
    #             else:
    #                 left_x = x1 + 10
    #                 right_x = x2 - 10
    #                 top_y = y1 + 10
    #                 bottom_y = y2 - 10
                    
    #                 output += f"{current_id}. Action from left to right: swipe, {left_x}, {middle_y}, {right_x}, {middle_y}\n"
    #                 current_id += 1
    #                 output += f"{current_id}. Action from right to left: swipe, {right_x}, {middle_y}, {left_x}, {middle_y}\n"
    #                 current_id += 1
    #                 output += f"{current_id}. Action from top to bottom: swipe, {middle_x}, {top_y}, {middle_x}, {bottom_y}\n"
    #                 current_id += 1
    #                 output += f"{current_id}. Action from bottom to top: swipe, {middle_x}, {bottom_y}, {middle_x}, {top_y}\n"
    #                 current_id += 1

    #         output += "\n"
    
    # if len(elements_dict['editable']) != 0:
    #     output += "=== EDITABLE ELEMENTS ===\n"
    #     for elem in elements_dict['editable']:
    #         # output += f"{elem['text'] or elem['desc'] or elem['resource_id']} ({elem['class']})\n"
    #         # output += f"{elem['display_text']} ({elem['class']})\n"
    #         output += f"{elem['display_text']} ({elem['class']})\n"
    #         output += f"Bounds: {elem['bounds']}\n"
    #         output += f"{current_id}. Action: input_text\n"
    #         current_id += 1
    #         output += "\n"
    
    # output += "=== UI-Independent Actions ===\n"
    # output += "\"Navigate to home screen\"\n"
    # output += f"{current_id}. Action: navigate_home\n"
    # output += "\n"
    # current_id += 1
    # output += "\"Navigate back to last screen\"\n"
    # output += f"{current_id}. Action: navigate_back\n"
    # output += "\n"
    # current_id += 1
    # output += "\"Wait for some time\"\n"
    # output += f"{current_id}. Action: wait, <duration>\n"
    # output += "\n"
    # current_id += 1
    # output += "\"Open an application with the app_name\"\n"
    # output += f"{current_id}. Action: open_app_with_name, <app_name>\n"
    
    return output
    # Note: omit complete task, answer users’ question for now

# Get XML string from file or other source
# xml_string = ...  # Replace with actual XML string

# Parse and extract UI elements
# elements = parse_xml_hierarchy(xml_string)

# Display the extracted elements
# display_ui_elements(elements)

# Example of using this extractor:
def rule_based_extractor(filepath):
    with open(filepath, 'r') as file:
        xml_string = file.read()
        elements = parse_xml_hierarchy(xml_string)
        ret = display_ui_elements(elements)
    return ret

if __name__ == "__main__":
    action_space = rule_based_extractor(filepath='/home/ubuntu/ninja-android-agent/10-3-69-24-5555/xml/window_dump.xml')
    print(action_space)