/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   I N S T A N C E S   C L A S S                                                                              */
/*                                                                                                              */ 

/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "instances.h"

namespace OpenNN
{


// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates a instances object with zero instances. 
/// It also initializes the rest of class members to their default values.

Instances::Instances()
{
   set();
}


// INSTANCES NUMBER CONSTRUCTOR

/// Instances number constructor. 
/// It creates a data set object with a given number of instances.
/// It also initializes the rest of class members to their default values.
/// @param new_instances_number Number of instances in the data set.

Instances::Instances(const size_t& new_instances_number)
{
    set(new_instances_number);
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates a instances object by loading the object members from a TinyXML document.
/// @param instances_document XML document from the TinyXML library.

Instances::Instances(const tinyxml2::XMLDocument& instances_document)
{   
   set(instances_document);
}


// COPY CONSTRUCTOR

/// Copy constructor. 
/// It creates a copy of an existing instances object. 
/// @param other_instances Instances object to be copied.

Instances::Instances(const Instances& other_instances)
{
   items = other_instances.items;

   display = other_instances.display;
}


// DESTRUCTOR

/// Destructor. 

Instances::~Instances()
{
}


// ASSIGNMENT OPERATOR

/// Assignment operator. 
/// It assigns to the current object the members of an existing instances object.
/// @param other_instances Instances object to be assigned.

Instances& Instances::operator= (const Instances& other_instances)
{
   if(this != &other_instances) 
   {
      items = other_instances.items;
      display = other_instances.display;
   }

   return(*this);
}


// EQUAL TO OPERATOR


/// Equal to operator. 
/// It compares this object with another object of the same class. 
/// It returns true if the members of the two objects have the same values, and false otherwise.
/// @ param other_instances Instances object to be compared with.

bool Instances::operator == (const Instances& other_instances) const
{
   if(/*items == other_instances.items
   &&*/ display == other_instances.display)
   {
      return(true);   
   }
   else
   {
      return(false);
   }
}


// METHODS


/// Returns a value of the splitting method enumeration from a string with that method.
/// @param splitting_method String with the method for splitting the instances.

Instances::SplittingMethod Instances::get_splitting_method(const string& splitting_method)
{
    if(splitting_method == "Sequential")
    {
        return(Sequential);
    }
    else if(splitting_method == "Random")
    {
        return(Random);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Instances class.\n"
               << "static SplittingMethod get_splitting_method(const string&).\n"
               << "Unknown splitting method: " << splitting_method << ".\n";

        throw logic_error(buffer.str());
    }
}


/// Returns empty if the number of instances is zero, and false otherwise.

bool Instances::empty() const
{
    if(items.empty())
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


/// Returns the use of every instance(training, selection, testing) in a vector. 

Vector<Instances::Use> Instances::get_uses() const
{
   const size_t instances_number = get_instances_number();

   Vector<Use> uses(instances_number);

   for(size_t i = 0; i < instances_number; i++)
   {
       uses[i] = items[i].use;
   }

   return(uses);
}


/// Returns the use of every instance(training, selection, testing) in a string vector. 

Vector<string> Instances::write_uses() const
{
   const size_t instances_number = get_instances_number();

   const Vector<Use> uses = get_uses();

   Vector<string> uses_string(instances_number);

   for(size_t i = 0; i < instances_number; i++)
   {
      if(uses[i] == Training)
	  {	  
	     uses_string[i] = "Training";    	     
	  }
      else if(uses[i] == Selection)
	  {	  
	     uses_string[i] = "Selection";    
	  }
      else if(uses[i] == Testing)
	  {	  
	     uses_string[i] = "Testing";    
	  }
      else if(uses[i] == Unused)
      {
         uses_string[i] = "Unused";
      }
      else
	  {
         ostringstream buffer;

         buffer << "OpenNN Exception Instances class.\n"
                << "Vector<string> write_uses() const method.\n"
                << "Unknown use.\n";
 
	     throw logic_error(buffer.str());
	  }
   }

   return(uses_string);
}


/// Returns the use of every instance(training, selection, testing) in a string vector.

Vector<string> Instances::write_abbreviated_uses() const
{
   const size_t instances_number = get_instances_number();

   const Vector<Use> uses = get_uses();

   Vector<string> uses_string(instances_number);

   for(size_t i = 0; i < instances_number; i++)
   {
      if(uses[i] == Training)
      {
         uses_string[i] = "Train.";
      }
      else if(uses[i] == Selection)
      {
         uses_string[i] = "Sel.";
      }
      else if(uses[i] == Testing)
      {
         uses_string[i] = "Test.";
      }
      else if(uses[i] == Unused)
      {
         uses_string[i] = "Unused";
      }
      else
      {
         ostringstream buffer;

         buffer << "OpenNN Exception Instances class.\n"
                << "Vector<string> write_abbreviated_uses() const method.\n"
                << "Unknown use.\n";

         throw logic_error(buffer.str());
      }
   }

   return(uses_string);
}


/// Returns the use of a single instance.
/// @param i Instance index.

const Instances::Use& Instances::get_use(const size_t& i) const
{
    return(items[i].use);
}


/// Returns a string the use name of a single instance.
/// @param i Instance index.

string Instances::write_use(const size_t& i) const
{
    if(items[i].use == Training)
    {
       return("Training");
    }
    else if(items[i].use == Selection)
    {
       return("Selection");
    }
    else if(items[i].use == Testing)
    {
       return("Testing");
    }
    else if(items[i].use == Unused)
    {
       return("Unused");
    }
    else
    {
       ostringstream buffer;

       buffer << "OpenNN Exception Instances class.\n"
              << "string write_use(const size_t&) const method.\n"
              << "Unknown use.\n";

       throw logic_error(buffer.str());
    }
}


/// Returns true if a given instance is to be used for training, selection or testing,
/// and false if it is to be unused.

bool Instances::is_used(const size_t& i) const
{
    if(items[i].use == Unused)
    {
        return(false);
    }
    else
    {
        return(true);
    }
}


/// Returns true if a given instance is to be unused and false in other case.

bool Instances::is_unused(const size_t& i) const
{
    if(items[i].use == Unused)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


/// Returns the number of instances in the data set which will neither be used
/// for training, selection or testing.

size_t Instances::get_unused_instances_number() const
{
    const size_t instances_number = get_instances_number();

    size_t unused_instances_number = 0;

    for(size_t i = 0; i < instances_number; i++)
	{
       if(items[i].use == Unused)
	   {
	      unused_instances_number++;
	   }
	}

	return(unused_instances_number);
}


/// Returns the total number of training, selection and testing instances,
/// i.e. those which are not "Unused".

size_t Instances::get_used_instances_number() const
{
    const size_t instances_number = get_instances_number();
    const size_t unused_instances_number = get_unused_instances_number();

    return(instances_number - unused_instances_number);
}


/// Returns the number of instances in the data set which will be used for training.

size_t Instances::get_training_instances_number() const
{
    const size_t instances_number = get_instances_number();

    size_t training_instances_number = 0;

    for(size_t i = 0; i < instances_number; i++)
	{
       if(items[i].use == Training)
	   {
	      training_instances_number++;
	   }
	}

	return(training_instances_number);
}


/// Returns the number of instances in the data set which will be used for selection.

size_t Instances::get_selection_instances_number() const
{
    const size_t instances_number = get_instances_number();

    size_t selection_instances_number = 0;

    for(size_t i = 0; i < instances_number; i++)
	{
       if(items[i].use == Selection)
	   {
	      selection_instances_number++;
	   }
	}

	return(selection_instances_number);
}


/// Returns the number of instances in the data set which will be used for testing.

size_t Instances::get_testing_instances_number() const
{
    const size_t instances_number = get_instances_number();

    size_t testing_instances_number = 0;

    for(size_t i = 0; i < instances_number; i++)
	{
       if(items[i].use == Testing)
	   {
	      testing_instances_number++;
	   }
	}

	return(testing_instances_number);
}


/// Returns a vector with the number of training, selection, testing
/// and unused instances.
/// The size of that vector is therefore four.

Vector<size_t> Instances::count_uses() const
{
    Vector<size_t> count(4, 0);

    const size_t instances_number = get_instances_number();

    for(size_t i = 0; i < instances_number; i++)
    {
        if(items[i].use == Training)
        {
           count[0]++;
        }
        else if(items[i].use == Selection)
        {
           count[1]++;
        }
        else if(items[i].use == Testing)
        {
           count[2]++;
        }
        else
        {
           count[3]++;
        }
    }

    return(count);
}


/// Returns the indices of the used instances(those which are not set unused).

Vector<size_t> Instances::get_used_indices() const
{
    const size_t instances_number = get_instances_number();

    const size_t used_instances_number = instances_number - get_unused_instances_number();

    Vector<size_t> used_indices(used_instances_number);

    size_t index = 0;

    for(size_t i = 0; i < instances_number; i++)
    {
        if(items[i].use != Unused)
        {
            used_indices[index] = i;
            index++;
        }
    }

    return(used_indices);
}


/// Returns the indices of the instances set unused.

Vector<size_t> Instances::get_unused_indices() const
{
    const size_t instances_number = get_instances_number();

    const size_t unused_instances_number = get_unused_instances_number();

    Vector<size_t> unused_indices(unused_instances_number);

    size_t count = 0;

    for(size_t i = 0; i < instances_number; i++)
    {
        if(items[i].use == Unused)
        {
            unused_indices[count] = static_cast<size_t>(i);
            count++;
        }
    }

    return(unused_indices);
}


/// Returns the indices of the instances which will be used for training.

Vector<size_t> Instances::get_training_indices() const
{
   const size_t instances_number = get_instances_number();

   const size_t training_instances_number = get_training_instances_number();

   Vector<size_t> training_indices(training_instances_number);

   size_t count = 0;

   for(size_t i = 0; i < instances_number; i++)
   {
      if(items[i].use == Training)
	  {
         training_indices[count] = static_cast<size_t>(i);
		 count++;
	  }
   }

   return(training_indices);
}


/// Returns the indices of the instances which will be used for selection.

Vector<size_t> Instances::get_selection_indices() const
{
   const size_t instances_number = get_instances_number();

   const size_t selection_instances_number = get_selection_instances_number();

   Vector<size_t> selection_indices(selection_instances_number);

   size_t count = 0;

   for(size_t i = 0; i < instances_number; i++)
   {
      if(items[i].use == Selection)
	  {
         selection_indices[count] = i;
		 count++;
	  }
   }

   return(selection_indices);
}


/// Returns the indices of the instances which will be used for testing.

Vector<size_t> Instances::get_testing_indices() const
{
   const size_t instances_number = get_instances_number();

   const size_t testing_instances_number = get_testing_instances_number();

   Vector<size_t> testing_indices(testing_instances_number);

   size_t count = 0;

   for(size_t i = 0; i < instances_number; i++)
   {
      if(items[i].use == Testing)
	  {
         testing_indices[count] = i;
		 count++;
	  }
   }

   return(testing_indices);
}


/// Returns the indices of the instances which will be used for training.

Vector<int> Instances::get_training_indices_int() const
{
   const size_t instances_number = get_instances_number();

   const size_t training_instances_number = get_training_instances_number();

   Vector<int> training_indices(training_instances_number);

   size_t count = 0;

   for(size_t i = 0; i < instances_number; i++)
   {
      if(items[i].use == Training)
      {
         training_indices[count] = static_cast<int>(i);
         count++;
      }
   }

   return(training_indices);
}


/// Returns the indices of the instances which will be used for selection.

Vector<int> Instances::get_selection_indices_int() const
{
   const size_t instances_number = get_instances_number();

   const size_t selection_instances_number = get_selection_instances_number();

   Vector<int> selection_indices(selection_instances_number);

   size_t count = 0;

   for(size_t i = 0; i < instances_number; i++)
   {
      if(items[i].use == Selection)
      {
         selection_indices[count] = static_cast<int>(i);
         count++;
      }
   }

   return(selection_indices);
}


/// Returns true if messages from this class can be displayed on the screen,
/// or false if messages from this class can't be displayed on the screen.

const bool& Instances::get_display() const
{
   return(display);   
}


/// Sets a instances object with zero instances. 

void Instances::set()
{
   set_instances_number(0);

   set_default();
}


/// Sets a new number of instances in the instances object. 
/// All the instances are set for training. 
/// @param new_instances_number Number of instances.

void Instances::set(const size_t& new_instances_number)
{
   set_instances_number(new_instances_number);

   set_default();
}


/// Sets the instances information members from a XML document.
/// @param instances_document Pointer to a TinyXML document containing the member data.

void Instances::set(const tinyxml2::XMLDocument& instances_document)
{
    set_default();

   from_XML(instances_document);
}


/// Sets the default values to this instances object:
/// <ul>
/// <li>display: true</li>
/// </ul>

void Instances::set_default()
{
    display = true;
}


/// Sets new uses to all the instances from a single vector.
/// @param new_uses Vector of use structures.
/// The size must be equal to the number of instances.

void Instances::set_uses(const Vector<Instances::Use>& new_uses)
{
    const size_t instances_number = get_instances_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__   

   const size_t new_uses_size = new_uses.size();

   if(new_uses_size != instances_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Instances class.\n"
             << "void set_uses(const Vector<Use>&) method.\n"
             << "Size of uses(" << new_uses_size << ") must be equal to number of instances(" << instances_number << ").\n";

	  throw logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < instances_number; i++)
   {
       items[i].use = new_uses[i];
   }
}


/// Sets new uses to all the instances from a single vector of strings.
/// @param new_uses Vector of use strings.
/// Possible values for the elements are "Training", "Selection", "Testing" and "Unused".
/// The size must be equal to the number of instances.

void Instances::set_uses(const Vector<string>& new_uses)
{
    const size_t instances_number = get_instances_number();

    ostringstream buffer;

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__   

   const size_t new_uses_size = new_uses.size();

   if(new_uses_size != instances_number)
   {
      buffer << "OpenNN Exception: Instances class.\n"
             << "void set_uses(const Vector<string>&) method.\n"
             << "Size of uses(" << new_uses_size << ") must be equal to number of instances(" << instances_number << ").\n";

	  throw logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < instances_number; i++)
   {   
      if(new_uses[i] == "Unused")
	  {	  
         items[i].use = Unused;
	  }
      else if(new_uses[i] == "Training")
	  {	  
         items[i].use = Training;
	  }
      else if(new_uses[i] == "Selection")
	  {	  
         items[i].use = Selection;
	  }
      else if(new_uses[i] == "Testing")
	  {	  
         items[i].use = Testing;
	  }
	  else
	  {
         buffer << "OpenNN Exception Instances class.\n"
                << "void set_uses(const Vector<string>&) method.\n"
                << "Unknown use.\n";
 
	     throw logic_error(buffer.str());
	  }
   }   
}


/// Sets the use of a single instance.
/// @param i Index of instance.
/// @param new_use Use for that instance.

void Instances::set_use(const size_t& i, const Use& new_use)
{
    items[i].use = new_use;
}


/// Sets the use of a single instance from a string.
/// @param i Index of instance.
/// @param new_use String with the use name("Training", "Selection", "Testing" or "Unused")

void Instances::set_use(const size_t& i, const string& new_use)
{
    if(new_use == "Training")
    {
       items[i].use = Training;
    }
    else if(new_use == "Selection")
    {
       items[i].use = Selection;
    }
    else if(new_use == "Testing")
    {
       items[i].use = Testing;
    }
    else if(new_use == "Unused")
    {
       items[i].use = Unused;
    }
    else
    {
       ostringstream buffer;

       buffer << "OpenNN Exception Instances class.\n"
              << "void set_use(const string&) method.\n"
              << "Unknown use: " << new_use << "\n";

       throw logic_error(buffer.str());
    }
}


/// Sets instances with given indices in the data set unused.

void Instances::set_unused(const Vector<size_t> & indices)
{
    for(int i = 0; i < static_cast<int>(indices.size()); i++)
    {
        const size_t index = indices[static_cast<size_t>(i)];

        items[index].use = Unused;
    }
}


/// Sets all the instances in the data set for training. 

void Instances::set_training()
{
   const size_t instances_number = get_instances_number();

   for(size_t i = 0; i < instances_number; i++)
   {
       items[i].use = Training;
   }
}


/// Sets all the instances in the data set for selection. 

void Instances::set_selection()
{
    const size_t instances_number = get_instances_number();

    for(size_t i = 0; i < instances_number; i++)
    {
        items[i].use = Selection;
    }
}


/// Sets all the instances in the data set for testing. 

void Instances::set_testing()
{
    const size_t instances_number = get_instances_number();

    for(size_t i = 0; i < instances_number; i++)
    {
        items[i].use = Testing;
    }
}


void Instances::set_unused()
{
    const size_t instances_number = get_instances_number();

    for(size_t i = 0; i < instances_number; i++)
    {
        items[i].use = Unused;
    }
}


void Instances::set_training(const Vector<size_t>& indices)
{
    size_t index = 0;

    for(size_t i = 0; i < indices.size(); i++)
    {
        index = indices[i];

        items[index].use = Training;
    }
}


void Instances::set_selection(const Vector<size_t>& indices)
{
    size_t index = 0;

    for(size_t i = 0; i < indices.size(); i++)
    {
        index = indices[i];

        items[index].use = Selection;
    }
}


void Instances::set_testing(const Vector<size_t>& indices)
{
    size_t index = 0;

    for(size_t i = 0; i < indices.size(); i++)
    {
        index = indices[i];

        items[index].use = Testing;
    }
}


void Instances::unuse_first_instances(const size_t& number)
{
    for(size_t i = 0; i < number; i++)
    {
        items[i].use = Unused;
    }
}


void Instances::set_training(const size_t& start, const size_t& end)
{
    for(size_t i = start; i < end; i++)
    {
        items[i].use = Training;
    }
}


void Instances::set_selection(const size_t& start, const size_t& end)
{
    for(size_t i = start; i < end; i++)
    {
        items[i].use = Selection;
    }
}


void Instances::set_testing(const size_t& start, const size_t& end)
{
    for(size_t i = start; i < end; i++)
    {
        items[i].use = Testing;
    }

}


void Instances::set_training_last(const size_t& number)
{
    const size_t instances_number = get_instances_number();

    const size_t start = instances_number - number;

    for(size_t i = start; i < instances_number; i++)
    {
        items[i].use = Training;
    }
}


void Instances::set_selection_last(const size_t& number)
{
    const size_t instances_number = get_instances_number();

    const size_t start = instances_number - number;

    for(size_t i = start; i < instances_number; i++)
    {
        items[i].use = Selection;
    }
}


void Instances::set_testing_last(const size_t& number)
{
    const size_t instances_number = get_instances_number();

    const size_t start = instances_number - number;

    for(size_t i = start; i < instances_number; i++)
    {
        items[i].use = Testing;
    }

}


void Instances::set_k_fold_cross_validation_uses(const size_t& k, const size_t& fold_index)
{
    const size_t instances_number = get_instances_number();

    const size_t fold_size = instances_number/k;

    const size_t start = fold_index*fold_size;
    const size_t end = start + fold_size;

    split_random_indices(1,0,0);

//    cout << "Start: " << start << endl;
//    cout << "End: " << end << endl;

    for(size_t i = start; i < end; i++)
    {
        items[i] = Testing;
    }
}


void Instances::selection_to_testing()
{
    const size_t instances_number = get_instances_number();

    for(size_t i = 0; i < instances_number; i++)
    {
        if(items[i].use == Selection) items[i].use = Testing;
    }
}

void Instances::testing_to_selection()
{
    const size_t instances_number = get_instances_number();

    for(size_t i = 0; i < instances_number; i++)
    {
        if(items[i].use == Testing) items[i].use = Selection;
    }
}


/// Sets a new display value. 
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void Instances::set_display(const bool& new_display)
{
   display = new_display;
}


/// Sets a new number of instances in the data set. 
/// All instances are also set for training. 
/// @param new_instances_number Number of instances. 

void Instances::set_instances_number(const size_t& new_instances_number)
{
   items.set(new_instances_number);

   split_instances();
}


/// Serializes the instances object into a XML document of the TinyXML library. 
/// See the OpenNN manual for more information about the format of this document. 

tinyxml2::XMLDocument* Instances::to_XML() const
{
   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   ostringstream buffer;

   // Instances

   tinyxml2::XMLElement* instances_element = document->NewElement("Instances");

   document->InsertFirstChild(instances_element);

   tinyxml2::XMLElement* element = nullptr;
   tinyxml2::XMLText* text = nullptr;

   const size_t instances_number = get_instances_number();

   // Instances number
   {
      element = document->NewElement("InstancesNumber");
      instances_element->LinkEndChild(element);

      buffer.str("");
      buffer << instances_number;

      text = document->NewText(buffer.str().c_str());
      element->LinkEndChild(text);
   }

   // Items uses

   element = document->NewElement("ItemsUses");
   instances_element->LinkEndChild(element);

   string items_uses;

   for(size_t i = 0; i < instances_number; i++)
   {
/*
       element = document->NewElement("Item");
       element->SetAttribute("Index",(unsigned)i+1);
       instances_element->LinkEndChild(element);

       // Use

       tinyxml2::XMLElement* use_element = document->NewElement("Use");
       element->LinkEndChild(use_element);

       tinyxml2::XMLText* use_text = document->NewText(write_use(i).c_str());
       use_element->LinkEndChild(use_text);
*/

       string item_use_string(number_to_string(static_cast<size_t>(get_use(i))));

       items_uses.append(item_use_string);

       if(i != instances_number - 1)
       {
           items_uses.append(" ");
       }
   }

   tinyxml2::XMLText* items_uses_text = document->NewText(items_uses.c_str());
   element->LinkEndChild(items_uses_text);

   // Display
//   {
//      element = document->NewElement("Display");
//      instances_element->LinkEndChild(element);

//      buffer.str("");
//      buffer << display;

//      text = document->NewText(buffer.str().c_str());
//      element->LinkEndChild(text);
//   }

   return(document);
}


/// Serializes the instances object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void Instances::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    const size_t instances_number = get_instances_number();

    file_stream.OpenElement("Instances");

    // Instances number

    file_stream.OpenElement("InstancesNumber");

    buffer.str("");
    buffer << instances_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Items uses

    string items_uses;

    for(size_t i = 0; i < instances_number; i++)
    {
        string item_use_string(number_to_string(static_cast<size_t>(get_use(i))));

        items_uses.append(item_use_string);

        if(i != instances_number - 1)
        {
            items_uses.append(" ");
        }
    }

    file_stream.OpenElement("ItemsUses");

    file_stream.PushText(items_uses.c_str());

    file_stream.CloseElement();


    file_stream.CloseElement();
}


/// Deserializes a TinyXML document into this instances object.
/// @param instances_document XML document containing the member data.

void Instances::from_XML(const tinyxml2::XMLDocument& instances_document)
{
    ostringstream buffer;

    // Instances  element

   const tinyxml2::XMLElement* instances_element = instances_document.FirstChildElement("Instances");

   if(!instances_element)
   {
      buffer << "OpenNN Exception: Instances class.\n"
             << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
             << "Pointer to instances information element is nullptr.\n";

	  throw logic_error(buffer.str());   
   }

   // Instances number

   const tinyxml2::XMLElement* instances_number_element = instances_element->FirstChildElement("InstancesNumber");

   if(!instances_number_element)
   {
      buffer << "OpenNN Exception: Instances class.\n"
             << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
             << "Pointer to instances number is nullptr.\n";

      throw logic_error(buffer.str());
   }

   const size_t instances_number = static_cast<size_t>(atoi(instances_number_element->GetText()));

   set_instances_number(instances_number);

   if(instances_number <= 0)
   {
       return;
   }

   // Items uses
/*
   unsigned index = 0;

   const tinyxml2::XMLElement* start_element = instances_number_element;

   for(size_t i = 0; i < instances_number; i++)
   {
       const tinyxml2::XMLElement* item_element = start_element->NextSiblingElement("Item");
       start_element = item_element;

      if(!item_element)
      {
          buffer << "OpenNN Exception: Instances class.\n"
                 << "void from_XML(const tinyxml2::XMLElement*) method.\n"
                 << "Item " << i+1 << " is nullptr.\n";

          throw logic_error(buffer.str());
      }

     item_element->QueryUnsignedAttribute("Index", &index);

     if(index != i+1)
     {
         buffer << "OpenNN Exception: Instances class.\n"
                << "void from_XML(const tinyxml2::XMLElement*) method.\n"
                << "Index " << index << " is not correct.\n";

         throw logic_error(buffer.str());
     }

     // Use

     const tinyxml2::XMLElement* use_element = item_element->FirstChildElement("Use");

     if(!use_element)
     {
        buffer << "OpenNN Exception: Instances class.\n"
               << "void from_XML(const tinyxml2::XMLElement*) method.\n"
               << "Pointer to use element is nullptr.\n";

        throw logic_error(buffer.str());
     }

     if(use_element->GetText())
     {
        set_use(index-1, use_element->GetText());
     }
  }
*/

   const tinyxml2::XMLElement* items_uses_element = instances_element->FirstChildElement("ItemsUses");

   if(!items_uses_element)
   {
       const tinyxml2::XMLElement* unknown_element = instances_element->FirstChildElement("Item");

       if(unknown_element)
       {
           buffer << "Project file is old.\n";
       }

      buffer << "OpenNN Exception: Instances class.\n"
             << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
             << "Pointer to items uses is nullptr.\n";

      throw logic_error(buffer.str());
   }

   // Read uses

   const string items_uses_text(items_uses_element->GetText());

   Vector<size_t> items_uses;
   items_uses.parse(items_uses_text);

   // Check instances number error

   if(instances_number != items_uses.size())
   {
       buffer << "OpenNN Exception: Instances class.\n"
              << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
              << "Items uses size(" << items_uses.size() << ") must be the same than instances number(" << instances_number << ").\n";

       throw logic_error(buffer.str());
   }

   for(size_t i = 0; i < instances_number; i++)
   {
       set_use(i, static_cast<Use>(items_uses.at(i)));
   }
}


/// Creates new training, selection and testing indices at random.
/// @param training_instances_ratio Ratio of training instances in the data set.
/// @param selection_instances_ratio Ratio of selection instances in the data set.
/// @param testing_instances_ratio Ratio of testing instances in the data set.

void Instances::split_random_indices
(const double& training_instances_ratio, const double& selection_instances_ratio, const double& testing_instances_ratio)
{
   const size_t used_instances_number = get_used_instances_number();

   if(used_instances_number == 0)
   {
       return;
   }

   const double total_ratio = training_instances_ratio + selection_instances_ratio + testing_instances_ratio;

   // Get number of instances for training, selection and testing

   const size_t selection_instances_number = static_cast<size_t>(selection_instances_ratio*used_instances_number/total_ratio);
   const size_t testing_instances_number = static_cast<size_t>(testing_instances_ratio*used_instances_number/total_ratio);
   const size_t training_instances_number = used_instances_number - selection_instances_number - testing_instances_number;

   const size_t sum_instances_number = training_instances_number + selection_instances_number + testing_instances_number;

   if(sum_instances_number != used_instances_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Warning: Instances class.\n" 
                << "void split_random_indices(const double&, const double&, const double&) method.\n"
                << "Sum of numbers of training, selection and testing instances is not equal to number of used instances.\n";

      throw logic_error(buffer.str());
   }

   const size_t instances_number = get_instances_number();

   Vector<size_t> indices(0, 1, instances_number-1);
   random_shuffle(indices.begin(), indices.end());

   size_t i = 0;
   size_t index;

   // Training

   size_t count_training = 0;

   while(count_training != training_instances_number)
   {
      index = indices[i];

      if(items[index].use != Unused)
      {
        items[index].use = Training;
        count_training++;
      }

      i++;
   }

   // Selection 

   size_t count_selection = 0;

   while(count_selection != selection_instances_number)
   {
      index = indices[i];

      if(items[index].use != Unused)
      {
        items[index].use = Selection;
        count_selection++;
      }

      i++;
   }

   // Testing 

   size_t count_testing = 0;

   while(count_testing != testing_instances_number)
   {
      index = indices[i];

      if(items[index].use != Unused)
      {
            items[index].use = Testing;
            count_testing++;
      }

      i++;
   }
}


/// Creates new training, selection and testing indices with sequential indices.
/// @param training_instances_ratio Ratio of training instances in the data set.
/// @param selection_instances_ratio Ratio of selection instances in the data set.
/// @param testing_instances_ratio Ratio of testing instances in the data set.

void Instances::split_sequential_indices(const double& training_instances_ratio, const double& selection_instances_ratio, const double& testing_instances_ratio)
{
   const size_t used_instances_number = get_used_instances_number();

   if(used_instances_number == 0)
   {
       return;
   }

   const double total_ratio = training_instances_ratio + selection_instances_ratio + testing_instances_ratio;

   // Get number of instances for training, selection and testing

   const size_t selection_instances_number = static_cast<size_t>(selection_instances_ratio*used_instances_number/total_ratio);
   const size_t testing_instances_number = static_cast<size_t>(testing_instances_ratio*used_instances_number/total_ratio);
   const size_t training_instances_number = used_instances_number - selection_instances_number - testing_instances_number;

   const size_t sum_instances_number = training_instances_number + selection_instances_number + testing_instances_number;

   if(sum_instances_number != used_instances_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Warning: Instances class.\n" 
             << "void split_random_indices(const double&, const double&, const double&) method.\n"
             << "Sum of numbers of training, selection and testing instances is not equal to number of used instances.\n";
   
      throw logic_error(buffer.str());
   }

   size_t i = 0;

   // Training

   size_t count_training = 0;

   while(count_training != training_instances_number)
   {
      if(items[i].use != Unused)
      {
        items[i].use = Training;
        count_training++;
      }

      i++;
   }

   // Selection

   size_t count_selection = 0;

   while(count_selection != selection_instances_number)
   {
      if(items[i].use != Unused)
      {
        items[i].use = Selection;
        count_selection++;
      }

      i++;
   }

   // Testing

   size_t count_testing = 0;

   while(count_testing != testing_instances_number)
   {
      if(items[i].use != Unused)
      {
            items[i].use = Testing;
            count_testing++;
      }

      i++;
   }
}


Vector< Vector<size_t> > Instances::get_training_batches(const size_t& batch_size) const
{
    const Vector<size_t> training_indices = get_training_indices();

    return training_indices.split(batch_size);
}


Vector< Vector<size_t> > Instances::get_selection_batches(const size_t& batch_size) const
{
    const Vector<size_t> selection_indices = get_selection_indices();

    return selection_indices.split(batch_size);
}


Vector< Vector<size_t> > Instances::get_testing_batches(const size_t& batch_size) const
{
    const Vector<size_t> testing_indices = get_testing_indices();

    return testing_indices.split(batch_size);
}


/// Divides the instances into training, selection and testing subsets.
/// @param splitting_method Instances splitting method(Sequential or Random, Random by default).
/// @param training_ratio Ratio of training instances(0.6 by default).
/// @param selection_ratio Ratio of selection instances(0.2 by default).
/// @param testing_ratio Ratio of testing instances(0.2 by default).

void Instances::split_instances(const SplittingMethod& splitting_method, const double& training_ratio, const double& selection_ratio, const double& testing_ratio)
{

#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(training_ratio < 0)
    {
        buffer << "OpenNN Exception: Instances class.\n"
               << "void split_instances(const SplittingMethod&, const double&, const double&, const double&) method.\n"
               << "Training ratio is lower than zero.\n";

        throw logic_error(buffer.str());
    }

    if(selection_ratio < 0)
    {
        buffer << "OpenNN Exception: Instances class.\n"
               << "void split_instances(const SplittingMethod&, const double&, const double&, const double&) method.\n"
               << "Selection ratio is lower than zero.\n";

        throw logic_error(buffer.str());
    }

    if(testing_ratio < 0)
    {
        buffer << "OpenNN Exception: Instances class.\n"
               << "void split_instances(const SplittingMethod&, const double&, const double&, const double&) method.\n"
               << "Testing ratio is lower than zero.\n";

        throw logic_error(buffer.str());
    }

    if(training_ratio == 0.0 && selection_ratio == 0.0 && testing_ratio == 0.0)
    {
        buffer << "OpenNN Exception: Instances class.\n"
               << "void split_instances(const SplittingMethod&, const double&, const double&, const double&) method.\n"
               << "All training, selection and testing ratios are zero.\n";

        throw logic_error(buffer.str());
    }

#endif

    switch(splitting_method)
    {
        case Instances::Sequential:
        {
             split_sequential_indices(training_ratio, selection_ratio, testing_ratio);
        }
        break;

        case Instances::Random:
        {
             split_random_indices(training_ratio, selection_ratio, testing_ratio);
        }
        break;
    }
}


/// Returns the percentages of
/// unused, training, selection and testing instances, respectively.

Vector<double> Instances::calculate_uses_percentage() const
{
    const size_t instances_number = get_instances_number();

    Vector<double> uses_percentage(instances_number);

    const Vector<size_t> uses_count = count_uses();

    for(size_t i = 0; i < 4; i++)
    {
        uses_percentage[i] = 100.0*uses_count[i]/static_cast<double>(instances_number);
    }

    return(uses_percentage);
}


/// Converts the instances when the data set is to be used for time series prediction.
/// This method modifies the number of instances.
/// The new number of instances will be instances_number - lags_number.

void Instances::convert_time_series(const size_t& lags_number, const size_t& steps_ahead_number)
{
    const size_t instances_number = get_instances_number();

    if(instances_number < lags_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Instances class.\n"
               << "void convert_time_series(const size_t&).\n"
               << "Number of instances(" << instances_number << ") must be equal or greater than number of lags(" << lags_number << ").\n";

        throw logic_error(buffer.str());
    }

    const size_t new_instances_number = instances_number - lags_number - steps_ahead_number + 1;

    set(new_instances_number);
}


/// Returns a string representation of the current instances object. 

string Instances::object_to_string() const
{
   ostringstream buffer;

   buffer << "Instances object\n"
	      << "Instances number: " << get_instances_number() << "\n"
	      << "Training instances number: " << get_training_instances_number() << "\n"
	      << "Selection instances number: " << get_selection_instances_number() << "\n"
	      << "Testing instances number: " << get_testing_instances_number() << "\n"
          << "Uses: " << write_uses() << "\n";
          //<< "Display: " << display << "\n";

   return(buffer.str());
}


/// Prints to the screen information about the instances object.

void Instances::print() const
{
    cout << object_to_string();
}

}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2018 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
