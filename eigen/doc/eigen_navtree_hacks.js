
// generate a table of contents in the side-nav based on the h1/h2 tags of the current page.
function generate_autotoc() {
  var headers = $("h1, h2");
  if(headers.length > 1) {
    var toc = $("#side-nav").append('<div id="nav-toc" class="toc"><h3>Table of contents</h3></div>');
    toc = $("#nav-toc");
    var footer  = $("#nav-path");
    var footerHeight = footer.height();
    toc = toc.append('<ul></ul>');
    toc = toc.find('ul');
    var indices = new Array();
    indices[0] = 0;
    indices[1] = 0;

    var h1counts = $("h1").length;
    headers.each(function(i) {
      var current = $(this);
      var levelTag = current[0].tagName.charAt(1);
      if(h1counts==0)
        levelTag--;
      var cur_id = current.attr("id");

      indices[levelTag-1]+=1;  
      var prefix = indices[0];
      if (levelTag >1) {
        prefix+="."+indices[1];
      }
        
      // Uncomment to add number prefixes
      // current.html(prefix + "   " + current.html());
      for(var l = levelTag; l < 2; ++l){
          indices[l] = 0;
      }

      if(cur_id == undefined) {
        current.attr('id', 'title' + i);
        current.addClass('anchor');
        toc.append("<li class='level" + levelTag + "'><a id='link" + i + "' href='#title" +
                    i + "' title='" + current.prop("tagName") + "'>" + current.text() + "</a></li>");
      } else {
        toc.append("<li class='level" + levelTag + "'><a id='" + cur_id + "' href='#title" +
                    i + "' title='" + current.prop("tagName") + "'>" + current.text() + "</a></li>");
      }
    });
    resizeHeight();
  }
}


var global_navtree_object;

// Overloaded to remove links to sections/subsections
function getNode(o, po)
{
  po.childrenVisited = true;
  var l = po.childrenData.length-1;
  for (var i in po.childrenData) {
    var nodeData = po.childrenData[i];
    if((!nodeData[1]) ||  (nodeData[1].indexOf('#')==-1)) // <- we added this line
      po.children[i] = newNode(o, po, nodeData[0], nodeData[1], nodeData[2], i==l);
  }
}

/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
 */
// We need to override entire resizable just so we can change the height to account for the TOC.
function initResizable()
{
  var cookie_namespace = 'doxygen';
  var sidenav,navtree,content,header,collapsed,collapsedWidth=0,barWidth=6,desktop_vp=768,titleHeight;

  function readCookie(cookie)
  {
    var myCookie = cookie_namespace+"_"+cookie+"=";
    if (document.cookie) {
      var index = document.cookie.indexOf(myCookie);
      if (index != -1) {
        var valStart = index + myCookie.length;
        var valEnd = document.cookie.indexOf(";", valStart);
        if (valEnd == -1) {
          valEnd = document.cookie.length;
        }
        var val = document.cookie.substring(valStart, valEnd);
        return val;
      }
    }
    return 0;
  }

  function writeCookie(cookie, val, expiration)
  {
    if (val==undefined) return;
    if (expiration == null) {
      var date = new Date();
      date.setTime(date.getTime()+(10*365*24*60*60*1000)); // default expiration is one week
      expiration = date.toGMTString();
    }
    document.cookie = cookie_namespace + "_" + cookie + "=" + val + "; expires=" + expiration+"; path=/";
  }

  function resizeWidth()
  {
    var windowWidth = $(window).width() + "px";
    var sidenavWidth = $(sidenav).outerWidth();
    content.css({marginLeft:parseInt(sidenavWidth)+"px"});
    writeCookie('width',sidenavWidth-barWidth, null);
  }

  function restoreWidth(navWidth)
  {
    var windowWidth = $(window).width() + "px";
    content.css({marginLeft:parseInt(navWidth)+barWidth+"px"});
    sidenav.css({width:navWidth + "px"});
  }

  function resizeHeight()
  {  
    var headerHeight = header.outerHeight();
    var footerHeight = footer.outerHeight();
    var windowHeight = $(window).height() - headerHeight - footerHeight;
    //==========================================================================
    // MODIFICATION:
    // This small section is the only portion modified within initResizable().
    // The rest is copy-pasted from the doxygen-generated resize.js.
    //
    // Adjust nav height to make room for TOC.
    var toc = $("#nav-toc");
    var tocHeight = toc.height();
    var navHeight = windowHeight;
    // tocHeight is not always defined (e.g. if empty)
    if (tocHeight) {
      navHeight = windowHeight - tocHeight;
    }
   //==========================================================================
    
    content.css({height:windowHeight + "px"});
    navtree.css({height:navHeight + "px"});
    sidenav.css({height:windowHeight + "px"});
    
    var width=$(window).width();
    if (width!=collapsedWidth) {
      if (width<desktop_vp && collapsedWidth>=desktop_vp) {
        if (!collapsed) {
          collapseExpand();
        }
      } else if (width>desktop_vp && collapsedWidth<desktop_vp) {
        if (collapsed) {
          collapseExpand();
        }
      }
      collapsedWidth=width;
    }
    if (location.hash.slice(1)) {
      (document.getElementById(location.hash.slice(1))||document.body).scrollIntoView();
    }
  }

  function collapseExpand()
  {
    if (sidenav.width()>0) {
      restoreWidth(0);
      collapsed=true;
    }
    else {
      var width = readCookie('width');
      if (width>200 && width<$(window).width()) { restoreWidth(width); } else { restoreWidth(200); }
      collapsed=false;
    }
  }
  header  = $("#top");
  sidenav = $("#side-nav");
  content = $("#doc-content");
  navtree = $("#nav-tree");
  footer  = $("#nav-path");

  $(".side-nav-resizable").resizable({resize: function(e, ui) { resizeWidth(); } });
  $(sidenav).resizable({ minWidth: 0 });
  $(window).resize(function() { resizeHeight(); });
  var device = navigator.userAgent.toLowerCase();
  var touch_device = device.match(/(iphone|ipod|ipad|android)/);
  if (touch_device) { /* wider split bar for touch only devices */
    $(sidenav).css({ paddingRight:'20px' });
    $('.ui-resizable-e').css({ width:'20px' });
    $('#nav-sync').css({ right:'34px' });
    barWidth=20;
  }
  var width = readCookie('width');
  if (width) { restoreWidth(width); } else { resizeWidth(); }
  resizeHeight();
  var url = location.href;
  var i=url.indexOf("#");
  if (i>=0) window.location.hash=url.substr(i);
  var _preventDefault = function(evt) { evt.preventDefault(); };
  $("#splitbar").bind("dragstart", _preventDefault).bind("selectstart", _preventDefault);
  $(".ui-resizable-handle").dblclick(collapseExpand);
  $(window).on('load',resizeHeight);
}

// Overloaded to save the root node into global_navtree_object
function initNavTree(toroot,relpath)
{
  var o = new Object();
  global_navtree_object = o; // <- we added this line
  o.toroot = toroot;
  o.node = new Object();
  o.node.li = document.getElementById("nav-tree-contents");
  o.node.childrenData = NAVTREE;
  o.node.children = new Array();
  o.node.childrenUL = document.createElement("ul");
  o.node.getChildrenUL = function() { return o.node.childrenUL; };
  o.node.li.appendChild(o.node.childrenUL);
  o.node.depth = 0;
  o.node.relpath = relpath;
  o.node.expanded = false;
  o.node.isLast = true;
  o.node.plus_img = document.createElement("img");
  o.node.plus_img.src = relpath+"ftv2pnode.png";
  o.node.plus_img.width = 16;
  o.node.plus_img.height = 22;

  if (localStorageSupported()) {
    var navSync = $('#nav-sync');
    if (cachedLink()) {
      showSyncOff(navSync,relpath);
      navSync.removeClass('sync');
    } else {
      showSyncOn(navSync,relpath);
    }
    navSync.click(function(){ toggleSyncButton(relpath); });
  }

  navTo(o,toroot,window.location.hash,relpath);

  $(window).bind('hashchange', function(){
     if (window.location.hash && window.location.hash.length>1){
       var a;
       if ($(location).attr('hash')){
         var clslink=stripPath($(location).attr('pathname'))+':'+
                               $(location).attr('hash').substring(1);
         a=$('.item a[class$="'+clslink+'"]');
       }
       if (a==null || !$(a).parent().parent().hasClass('selected')){
         $('.item').removeClass('selected');
         $('.item').removeAttr('id');
       }
       var link=stripPath2($(location).attr('pathname'));
       navTo(o,link,$(location).attr('hash'),relpath);
     } else if (!animationInProgress) {
       $('#doc-content').scrollTop(0);
       $('.item').removeClass('selected');
       $('.item').removeAttr('id');
       navTo(o,toroot,window.location.hash,relpath);
     }
  })

  $(window).on("load", showRoot);
}

// return false if the the node has no children at all, or has only section/subsection children
function checkChildrenData(node) {
  if (!(typeof(node.childrenData)==='string')) {
    for (var i in node.childrenData) {
      var url = node.childrenData[i][1];
      if(url.indexOf("#")==-1)
        return true;
    }
    return false;
  }
  return (node.childrenData);
}

// Modified to:
// 1 - remove the root node 
// 2 - remove the section/subsection children
function createIndent(o,domNode,node,level)
{
  var level=-2; // <- we replaced level=-1 by level=-2
  var n = node;
  while (n.parentNode) { level++; n=n.parentNode; }
  if (checkChildrenData(node)) { // <- we modified this line to use checkChildrenData(node) instead of node.childrenData
    var imgNode = document.createElement("span");
    imgNode.className = 'arrow';
    imgNode.style.paddingLeft=(16*level).toString()+'px';
    imgNode.innerHTML=arrowRight;
    node.plus_img = imgNode;
    node.expandToggle = document.createElement("a");
    node.expandToggle.href = "javascript:void(0)";
    node.expandToggle.onclick = function() {
      if (node.expanded) {
        $(node.getChildrenUL()).slideUp("fast");
        node.plus_img.innerHTML=arrowRight;
        node.expanded = false;
      } else {
        expandNode(o, node, false, false);
      }
    }
    node.expandToggle.appendChild(imgNode);
    domNode.appendChild(node.expandToggle);
  } else {
    var span = document.createElement("span");
    span.className = 'arrow';
    span.style.width   = 16*(level+1)+'px';
    span.innerHTML = '&#160;';
    domNode.appendChild(span);
  }
}

// Overloaded to automatically expand the selected node
function selectAndHighlight(hash,n)
{
  var a;
  if (hash) {
    var link=stripPath($(location).attr('pathname'))+':'+hash.substring(1);
    a=$('.item a[class$="'+link+'"]');
  }
  if (a && a.length) {
    a.parent().parent().addClass('selected');
    a.parent().parent().attr('id','selected');
    highlightAnchor();
  } else if (n) {
    $(n.itemDiv).addClass('selected');
    $(n.itemDiv).attr('id','selected');
  }
  if ($('#nav-tree-contents .item:first').hasClass('selected')) {
    $('#nav-sync').css('top','30px');
  } else {
    $('#nav-sync').css('top','5px');
  }
  expandNode(global_navtree_object, n, true, true); // <- we added this line
  showRoot();
}


$(document).ready(function() {
  
  generate_autotoc();
  
  (function (){ // wait until the first "selected" element has been created
    try {
      // this line will trigger an exception if there is no #selected element, i.e., before the tree structure is
      // complete.
      document.getElementById("selected").className = "item selected";
      
      // ok, the default tree has been created, we can keep going...
      
      // expand the "Chapters" node
      if(window.location.href.indexOf('unsupported')==-1)
        expandNode(global_navtree_object, global_navtree_object.node.children[0].children[2], true, true);
      else
        expandNode(global_navtree_object, global_navtree_object.node.children[0].children[1], true, true);
      
      // Hide the root node "Eigen"
      $(document.getElementsByClassName('index.html')[0]).parent().parent().css({display:"none"});
      
    } catch (err) {
      setTimeout(arguments.callee, 10);
    }
  })();
});
